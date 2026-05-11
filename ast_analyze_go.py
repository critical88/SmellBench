"""
Go function-to-test mapping using aggregate coverage analysis.
"""
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class GoFunctionInfo:
    """Information about a Go function."""
    name: str
    filepath: Path
    start_line: int
    end_line: int
    receiver: str = None


def get_spec(project_name: str) -> dict:
    """Load project spec from repo_list.json."""
    repo_list_path = Path(__file__).parent / "repo_list.json"
    with open(repo_list_path) as f:
        repos = json.load(f)
    return repos.get(project_name)


def build_go_function_index(src_root: Path, package_prefix: str) -> Dict[str, GoFunctionInfo]:
    """Build index of exported functions in Go source files."""
    function_index = {}

    for go_file in src_root.rglob("*.go"):
        if go_file.name.endswith("_test.go"):
            continue

        try:
            content = go_file.read_text(encoding="utf-8")
        except Exception:
            continue

        pkg_match = re.search(r"^package\s+(\w+)", content, re.MULTILINE)
        if not pkg_match:
            continue
        pkg_name = pkg_match.group(1)

        func_pattern = re.compile(
            r"^func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(",
            re.MULTILINE,
        )
        lines = content.splitlines()

        for i, line in enumerate(lines, start=1):
            match = func_pattern.match(line)
            if not match:
                continue

            _, receiver_type, func_name = match.groups()
            if not func_name:
                continue

            end_line = find_function_end(lines, i - 1)
            if receiver_type:
                func_key = f"{pkg_name}.{receiver_type}.{func_name}"
            else:
                func_key = f"{pkg_name}.{func_name}"

            function_index[func_key] = GoFunctionInfo(
                name=func_name,
                filepath=go_file.resolve(),
                start_line=i,
                end_line=end_line,
                receiver=receiver_type,
            )

    return function_index


def find_function_end(lines: List[str], start_idx: int) -> int:
    """Find function end by brace counting."""
    depth = 0
    found_open = False

    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    return i + 1

    return start_idx + 1


def discover_test_functions(test_root: Path) -> List[str]:
    """Discover all top-level Go test functions."""
    test_functions = []
    test_pattern = re.compile(r"^func\s+(Test\w+)\s*\(", re.MULTILINE)

    for test_file in test_root.rglob("*_test.go"):
        try:
            content = test_file.read_text(encoding="utf-8")
        except Exception:
            continue

        test_functions.extend(match.group(1) for match in test_pattern.finditer(content))

    return sorted(set(test_functions))


def _go_env(project_root: Path) -> Dict[str, str]:
    """Get Go environment variables with custom cache paths."""
    env = os.environ.copy()
    env["GOCACHE"] = str((project_root / ".gocache").resolve())
    env["GOMODCACHE"] = str((project_root / ".gomodcache").resolve())
    return env


def cleanup_go_caches(project_root: Path):
    """Clean up Go cache directories with proper permission handling.

    NOTE: This function is intentionally disabled. Go caches (.gocache, .gomodcache)
    should be kept to speed up subsequent builds and tests. They are automatically
    managed by Go and don't need manual cleanup.
    """
    # Cache cleanup is disabled - Go caches speed up subsequent operations
    # and are automatically managed by Go itself
    pass


def run_tests_with_coverage(project_root: Path, coverage_file: Path, test_cmd: str = "") -> bool:
    """Run Go tests with aggregate coverage."""
    if coverage_file.exists():
        coverage_file.unlink()

    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    env = _go_env(project_root)

    # Download dependencies first (silent)
    # Note: No timeout set - let it take as long as needed for large dependencies
    subprocess.run(
        ["go", "mod", "download"],
        cwd=str(project_root),
        env=env,
        check=False,
    )

    cmd = [
        "go", "test",
        f"-coverprofile={coverage_file}",
        "-covermode=set",
        "-timeout=10m",
    ]
    if test_cmd:
        cmd.extend(test_cmd.split())
    cmd.append("./...")

    print(f"Running: {' '.join(cmd)}")
    # Don't capture output - let it stream to console like Java version
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        timeout=600,
        check=False,
    )

    if result.returncode != 0:
        print(f"Warning: go test exited with code {result.returncode}")

    if not coverage_file.exists():
        return False

    try:
        if coverage_file.read_text(encoding="utf-8").strip() == "mode: set":
            return False
    except Exception:
        return False

    return True


def parse_go_coverage(
    coverage_file: Path,
    src_root: Path,
    package_prefix: str,
) -> Dict[Path, Set[int]]:
    """Parse Go coverage profile into covered lines by absolute file path."""
    covered_lines: Dict[Path, Set[int]] = {}

    # Pre-build filename index to avoid repeated rglob calls
    print("Building filename index...")
    filename_index: Dict[str, List[Path]] = {}
    for go_file in src_root.rglob("*.go"):
        filename = go_file.name
        filename_index.setdefault(filename, []).append(go_file.resolve())
    print(f"Indexed {len(filename_index)} unique filenames")

    total_entries = 0
    covered_entries = 0
    skipped_entries = 0

    try:
        with coverage_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("mode:"):
                    continue

                total_entries += 1
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                location = parts[0]
                count = int(parts[2])
                if count == 0:
                    continue

                covered_entries += 1
                match = re.match(r"(.+):(\d+)\.\d+,(\d+)\.\d+", location)
                if not match:
                    continue

                file_path, start_line, end_line = match.groups()
                start_line = int(start_line)
                end_line = int(end_line)

                relative_path = None
                if package_prefix and file_path.startswith(package_prefix + "/"):
                    relative_path = file_path[len(package_prefix) + 1:]
                else:
                    parts_list = file_path.split("/")
                    if len(parts_list) > 3:
                        relative_path = "/".join(parts_list[3:])

                # Try direct path first
                actual_file = None
                if relative_path:
                    candidate = (src_root / relative_path).resolve()
                    if candidate.exists():
                        actual_file = candidate

                # Fallback: use filename index
                if actual_file is None:
                    filename = file_path.split("/")[-1]
                    candidates = filename_index.get(filename, [])
                    if candidates:
                        actual_file = candidates[0]  # Use first match

                if actual_file is None:
                    skipped_entries += 1
                    continue

                file_lines = covered_lines.setdefault(actual_file, set())
                for line_num in range(start_line, end_line + 1):
                    file_lines.add(line_num)
    except Exception as e:
        print(f"Error parsing coverage: {e}")

    print(f"Coverage parsing: {total_entries} total entries, {covered_entries} covered, {skipped_entries} skipped")
    return covered_lines


def is_function_covered(func_info: GoFunctionInfo, covered_lines: Dict[Path, Set[int]]) -> bool:
    """Check whether any line in the function range is covered."""
    file_covered = covered_lines.get(func_info.filepath.resolve())
    if not file_covered:
        return False

    return any(line_num in file_covered for line_num in range(func_info.start_line, func_info.end_line + 1))


def generate_go_function_mapping(
    project_name: str,
    project_path: str = "../project",
    output_dir: str = None,
    max_tests: int = None,
) -> int:
    """Generate function-to-test mapping for Go project."""
    repo_spec = get_spec(project_name)
    if not repo_spec:
        print(f"Project {project_name} not found")
        return 1

    if repo_spec.get("language") != "go":
        print(f"Project {project_name} is not a Go project")
        return 1

    project_root = (Path(project_path) / project_name).resolve()
    src_path = repo_spec.get("src_path", ".")
    commit_id = repo_spec.get("commit_id", "")
    package_prefix = repo_spec.get("package_prefix", "")
    test_cmd = repo_spec.get("test_cmd", "")
    src_root = (project_root / src_path).resolve()

    print(f"Project: {project_name}")
    print(f"Source root: {src_root}")

    print("Building function index...")
    function_index = build_go_function_index(src_root, package_prefix)
    print(f"Found {len(function_index)} functions")

    print("Discovering test functions...")
    test_functions = discover_test_functions(src_root)
    print(f"Found {len(test_functions)} test functions")

    print("Running tests with coverage...")
    coverage_file = (project_root / "coverage.out").resolve()
    if not run_tests_with_coverage(project_root, coverage_file, test_cmd=test_cmd):
        print("ERROR: Failed to generate non-empty coverage file")
        return 1

    print(f"Coverage file generated: {coverage_file}")
    print("Parsing coverage...")
    covered_lines = parse_go_coverage(coverage_file, src_root, package_prefix)
    print(f"Found coverage for {len(covered_lines)} files")

    print("Building function-to-test mapping...")
    functions = {}
    for func_key, func_info in function_index.items():
        if is_function_covered(func_info, covered_lines):
            functions[func_key] = {
                "file": str(func_info.filepath),
                "relative_file": str(func_info.filepath.relative_to(src_root)),
                "line_range": [func_info.start_line, func_info.end_line],
                "tests": test_functions,
                "coverage_type": "aggregate",
            }

    print(f"Mapped {len(functions)} functions to tests")
    if not functions:
        print("ERROR: No function coverage collected. Functions field is empty.")
        return 1

    if output_dir is None:
        output_dir = f"output/{project_name}"

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_json = output_dir_path / "function_testunit_mapping.json"

    payload = {
        "meta": {
            "src_path": src_path,
            "test_path": src_path,
            "commit_id": commit_id,
            "language": "go",
            "note": "Simplified mapping - all covered functions map to all tests",
        },
        "functions": functions,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote mapping to {output_json}")
    print(f"SUCCESS: {len(functions)} functions mapped")
    return 0


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "gin"
    sys.exit(generate_go_function_mapping(project, "../project", f"output/{project}"))
