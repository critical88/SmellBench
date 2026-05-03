"""
Diff Analyzer - Extract modified functions from git diffs
==========================================================

Analyzes unified diff output to determine which functions/methods were modified,
for both Python and Java projects. This provides more reliable test mapping than
relying on LLM-generated function lists.

Supports:
- Python: Uses AST to map diff lines to functions/methods/classes
- Java: Uses javalang to map diff lines to methods/constructors/classes
"""
import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Diff parsing (language-agnostic)
# ---------------------------------------------------------------------------

def parse_diff_changed_lines(diff_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse a unified diff and return per-file change info.

    Returns {file_path: {"new_lines": set[int], "hunk_ranges": list[tuple[int, int]]}}.
    - new_lines: individual added (+) line numbers in the new file.
    - hunk_ranges: (start, end) range of each hunk in the new file,
      covering the full span including context around additions and deletions.
    """
    changed: Dict[str, Dict[str, Any]] = {}
    current_file: Optional[str] = None
    new_line: Optional[int] = None

    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")

        if line.startswith("diff --git"):
            current_file = None
            new_line = None
            continue

        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                current_file = None
                continue
            if path.startswith("b/"):
                path = path[2:]
            current_file = path
            new_line = None
            if current_file not in changed:
                changed[current_file] = {"new_lines": set(), "hunk_ranges": []}
            continue

        if not current_file:
            continue

        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                new_start = int(match.group(1))
                new_count = int(match.group(2)) if match.group(2) else 1
                new_line = new_start
                if new_count > 0:
                    changed[current_file]["hunk_ranges"].append(
                        (new_start, new_start + new_count - 1)
                    )
            else:
                new_line = None
            continue

        if new_line is None:
            continue

        prefix = line[:1]
        if prefix == "+":
            changed[current_file]["new_lines"].add(new_line)
            new_line += 1
        elif prefix == "-":
            pass  # deletions don't advance new_line
        elif prefix == "\\":
            continue
        else:
            # Context line
            new_line += 1

    return changed


# ---------------------------------------------------------------------------
# Python AST-based function extraction
# ---------------------------------------------------------------------------

def _find_parent_class(tree: ast.AST, func_node: ast.FunctionDef) -> Optional[str]:
    """Walk the tree to find the parent class of a function, if any."""
    # Build parent map
    parent_map = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    # Walk up from func_node to find ClassDef
    current = func_node
    while current in parent_map:
        parent = parent_map[current]
        if isinstance(parent, ast.ClassDef):
            return parent.name
        current = parent
    return None


def extract_modified_functions_python(
    diff_content: str,
    repo_path: str,
    verbose: bool = False,
) -> List[Tuple[str, str, str]]:
    """Extract modified Python functions from a diff.

    Args:
        diff_content: Unified diff string
        repo_path: Repository root path
        verbose: If True, print debug information

    Returns:
        List of (file_path, class_name, function_name) tuples.
        class_name is empty string for top-level functions.
    """
    changed_files = parse_diff_changed_lines(diff_content)
    if verbose:
        print(f"  [DEBUG] Found {len(changed_files)} changed files in diff")
        for fpath in changed_files:
            print(f"    [DEBUG] {fpath}")

    modified_functions: List[Tuple[str, str, str]] = []

    for file_path, info in changed_files.items():
        if not file_path.endswith('.py'):
            if verbose:
                print(f"  [DEBUG] Skipping non-Python file: {file_path}")
            continue

        new_lines: Set[int] = info.get('new_lines', set())
        hunk_ranges: List[Tuple[int, int]] = info.get('hunk_ranges', [])

        if verbose:
            print(f"  [DEBUG] Analyzing {file_path}: {len(new_lines)} new lines, {len(hunk_ranges)} hunks")
            print(f"    [DEBUG] new_lines: {sorted(new_lines)[:10]}{'...' if len(new_lines) > 10 else ''}")

        if not new_lines and not hunk_ranges:
            if verbose:
                print(f"  [DEBUG] No changes in {file_path}, skipping")
            continue

        # Read source file
        abs_path = Path(repo_path) / file_path
        if verbose:
            print(f"  [DEBUG] Looking for file: {abs_path}")
            print(f"  [DEBUG] File exists: {abs_path.exists()}")
        if not abs_path.exists():
            if verbose:
                print(f"  [DEBUG] File not found: {abs_path}")
            continue

        try:
            source = abs_path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            if verbose:
                print(f"  [DEBUG] Successfully parsed {file_path}")
        except Exception as e:
            print(f"  Warning: failed to parse {file_path}: {e}")
            continue

        # Track which functions have been found to avoid duplicates
        seen = set()

        # Count functions for debugging
        func_count = 0

        # Walk AST to find modified functions
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_count += 1
            # Get function line range
            start = node.lineno
            end = node.end_lineno or node.lineno

            if verbose:
                class_name_debug = _find_parent_class(tree, node) or "(top-level)"
                print(f"  [DEBUG] Function {class_name_debug}.{node.name} at lines {start}-{end}")

            # Check if any changes overlap with this function
            line_hit = any(start <= ln <= end for ln in new_lines)
            hunk_hit = any(
                r_start <= end and r_end >= start
                for r_start, r_end in hunk_ranges
            )

            if verbose and (line_hit or hunk_hit):
                print(f"  [DEBUG]   -> HIT! line_hit={line_hit}, hunk_hit={hunk_hit}")

            if line_hit or hunk_hit:
                # Find parent class if any
                class_name = _find_parent_class(tree, node) or ""
                key = (file_path, class_name, node.name)
                if key not in seen:
                    modified_functions.append(key)
                    seen.add(key)

        if verbose:
            print(f"  [DEBUG] Total functions found in file: {func_count}")

        # Also check for modified classes (for class-level smells)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            start = node.lineno
            end = node.end_lineno or node.lineno

            line_hit = any(start <= ln <= end for ln in new_lines)
            hunk_hit = any(
                r_start <= end and r_end >= start
                for r_start, r_end in hunk_ranges
            )

            if line_hit or hunk_hit:
                # Add class entry (empty method name means class-level)
                key = (file_path, node.name, "")
                if key not in seen:
                    modified_functions.append(key)
                    seen.add(key)

    return modified_functions


# ---------------------------------------------------------------------------
# Java javalang-based function extraction
# ---------------------------------------------------------------------------

def extract_modified_functions_java(
    diff_content: str,
    repo_path: str,
    verbose: bool = False,
) -> List[Tuple[str, str, str]]:
    """Extract modified Java methods from a diff.

    Args:
        diff_content: Unified diff string
        repo_path: Repository root path
        verbose: If True, print debug information

    Returns:
        List of (file_path, class_name, method_name) tuples.
        class_name may include nested classes (e.g., "OuterClass.InnerClass").
    """
    if not JAVALANG_AVAILABLE:
        print("  Warning: javalang not available, cannot analyze Java diffs")
        return []

    changed_files = parse_diff_changed_lines(diff_content)
    if verbose:
        print(f"  [DEBUG] Found {len(changed_files)} changed files in diff")
        for fpath in changed_files:
            print(f"    [DEBUG] {fpath}")

    modified_functions: List[Tuple[str, str, str]] = []

    for file_path, info in changed_files.items():
        if not file_path.endswith('.java'):
            continue

        new_lines: Set[int] = info.get('new_lines', set())
        hunk_ranges: List[Tuple[int, int]] = info.get('hunk_ranges', [])

        if not new_lines and not hunk_ranges:
            continue

        # Read source file
        abs_path = Path(repo_path) / file_path
        if not abs_path.exists():
            continue

        try:
            source = abs_path.read_text(encoding='utf-8')
            tree = javalang.parse.parse(source)
        except Exception as e:
            print(f"  Warning: failed to parse {file_path}: {e}")
            continue

        source_lines = source.splitlines()
        seen = set()

        # Find modified methods
        for path_nodes, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.position is None:
                continue

            start_line = node.position.line
            # Estimate end line by counting braces
            end_line = _estimate_java_end_line(source_lines, start_line)

            # Check overlap
            line_hit = any(start_line <= ln <= end_line for ln in new_lines)
            hunk_hit = any(
                r_start <= end_line and r_end >= start_line
                for r_start, r_end in hunk_ranges
            )

            if line_hit or hunk_hit:
                # Build qualified class name (handles nested classes)
                class_scope = [
                    n.name for n in path_nodes
                    if hasattr(n, 'name') and isinstance(
                        n, (javalang.tree.ClassDeclaration,
                            javalang.tree.InterfaceDeclaration,
                            javalang.tree.EnumDeclaration))
                ]
                class_name = ".".join(class_scope) if class_scope else ""
                key = (file_path, class_name, node.name)
                if key not in seen:
                    modified_functions.append(key)
                    seen.add(key)

        # Find modified constructors
        for path_nodes, node in tree.filter(javalang.tree.ConstructorDeclaration):
            if node.position is None:
                continue

            start_line = node.position.line
            end_line = _estimate_java_end_line(source_lines, start_line)

            line_hit = any(start_line <= ln <= end_line for ln in new_lines)
            hunk_hit = any(
                r_start <= end_line and r_end >= start_line
                for r_start, r_end in hunk_ranges
            )

            if line_hit or hunk_hit:
                class_scope = [
                    n.name for n in path_nodes
                    if hasattr(n, 'name') and isinstance(
                        n, (javalang.tree.ClassDeclaration,
                            javalang.tree.InterfaceDeclaration,
                            javalang.tree.EnumDeclaration))
                ]
                class_name = ".".join(class_scope) if class_scope else ""
                # Constructor uses class name as method name
                key = (file_path, class_name, node.name)
                if key not in seen:
                    modified_functions.append(key)
                    seen.add(key)

        # Also check for modified classes (for class-level smells)
        for path_nodes, node in tree.filter(javalang.tree.ClassDeclaration):
            if node.position is None:
                continue

            start_line = node.position.line
            # For classes, scan until we find the matching closing brace
            end_line = _estimate_java_end_line(source_lines, start_line)

            line_hit = any(start_line <= ln <= end_line for ln in new_lines)
            hunk_hit = any(
                r_start <= end_line and r_end >= start_line
                for r_start, r_end in hunk_ranges
            )

            if line_hit or hunk_hit:
                class_scope = [
                    n.name for n in path_nodes
                    if hasattr(n, 'name') and isinstance(
                        n, (javalang.tree.ClassDeclaration,
                            javalang.tree.InterfaceDeclaration,
                            javalang.tree.EnumDeclaration))
                ]
                qualname = ".".join(class_scope + [node.name])
                # Class-level: empty method name
                key = (file_path, qualname, "")
                if key not in seen:
                    modified_functions.append(key)
                    seen.add(key)

    return modified_functions


def _estimate_java_end_line(source_lines: List[str], start_line: int) -> int:
    """Estimate end line of a Java method/class by counting braces."""
    depth = 0
    found_open = False
    for i in range(start_line - 1, len(source_lines)):
        line = source_lines[i]
        # Strip strings and comments to avoid counting braces inside them
        stripped = _strip_strings_and_comments(line)
        for ch in stripped:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    return i + 1  # 1-indexed
    return start_line


def _strip_strings_and_comments(line: str) -> str:
    """Roughly strip string literals and line comments from a line."""
    result = []
    i = 0
    in_string = False
    quote_char = None
    while i < len(line):
        ch = line[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == quote_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            i += 1
            continue
        if ch == '/' and i + 1 < len(line):
            if line[i + 1] == '/':
                break  # rest is comment
            if line[i + 1] == '*':
                # skip block comment on this line
                end = line.find('*/', i + 2)
                if end != -1:
                    i = end + 2
                    continue
                else:
                    break
        result.append(ch)
        i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def extract_modified_functions(
    diff_content: str,
    repo_path: str,
    language: str,
    verbose: bool = False,
) -> List[Tuple[str, str, str]]:
    """Extract modified functions from a diff (language-aware).

    Args:
        diff_content: Unified diff string
        repo_path: Repository root path
        language: "python" or "java"
        verbose: If True, print debug information

    Returns:
        List of (file_path, class_name, function_name) tuples.
    """
    language = language.lower()
    if language == "python":
        return extract_modified_functions_python(diff_content, repo_path, verbose=verbose)
    elif language == "java":
        return extract_modified_functions_java(diff_content, repo_path, verbose=verbose)
    else:
        raise ValueError(f"Unsupported language: {language}")


# ---------------------------------------------------------------------------
# Debugging utility
# ---------------------------------------------------------------------------

def print_modified_functions(functions: List[Tuple[str, str, str]]) -> None:
    """Pretty-print the list of modified functions."""
    if not functions:
        print("  No modified functions found")
        return

    print(f"  Found {len(functions)} modified function(s):")
    for file_path, class_name, func_name in functions:
        if class_name and func_name:
            label = f"{class_name}.{func_name}"
        elif class_name:
            label = f"{class_name} (class-level)"
        else:
            label = func_name or "(unknown)"
        print(f"    {file_path} :: {label}")
