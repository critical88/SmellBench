"""
Find Smell Candidates
=====================
Uses Claude Code CLI to find suitable method/class candidates for each
smell type in a repository. Results are cached to output/{repo_name}/candidates.json
so subsequent runs skip the agent call.

Usage:
    python find_candidates.py --project-name click
    python find_candidates.py --project-name pandas --force
    python find_candidates.py  # all selected repos
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from file_collector import collect_source_files
from claude_cli import call_claude_cli, extract_json_from_response


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_candidate_prompt(
    smell: Dict,
    eligible_files: List[Dict],
    src_path: str,
) -> str:
    """Build a prompt asking Claude to find 5 candidates for a single smell type.

    Strategy: provide the file list and let Claude use targeted grep/read
    on a few promising files rather than scanning everything.
    """
    files_desc = "\n".join(
        f"  - {f['file']} ({f['lines']} lines)" for f in eligible_files
    )

    smell_type = smell["type"]
    smell_desc = smell["desc"]
    hints = _smell_search_hints(smell_type)
    hints_line = f"\n**What to look for**: {hints}" if hints else ""

    return f"""You are a code analysis expert. Your task is to find methods/classes where it would be **convenient to inject** a specific code smell — NOT to find places that already exhibit this smell.

The goal is to identify code locations where the structure, complexity, and cross-module relationships make it natural and easy to introduce the smell while keeping the code compilable and tests passing.

## Source path: `{src_path}`

## Eligible files (lines > 500, no utility/helper files)
{files_desc}

## Smell type to inject: {smell_type}
{smell_desc}
{hints_line}

## Strategy — BE EFFICIENT
Do NOT read every file. Instead:
1. Based on file names and module structure, pick the 3-5 most promising files
2. Use `grep` to quickly locate class definitions, large methods, and cross-module interactions
3. Only read specific sections of files (use line ranges) to verify candidates
4. Prioritize files with core business logic (e.g., core.py, models.py, engine.py) over peripherals

## Requirements
Find exactly 5 candidates. Each candidate should be a method/class where injecting `{smell_type}` would be **easy and natural** — meaning the surrounding code structure supports the injection without breaking functionality.

Each candidate needs:
- `file`: relative path from repo root
- `class_name`: class name (null if standalone function)
- `method_name`: method/function name (for god_classes/interface_segregation, can be null)
- `line_number`: the actual starting line number (verify by reading)
- `reason`: 1-2 sentences explaining why this location is a good **injection point** (what structural properties make it easy to introduce the smell here)

**IMPORTANT**: At most 2 candidates may come from the same file. Spread candidates across different files to ensure diversity.

**DIVERSITY REQUIREMENT**: The 5 candidates must be substantially different from each other:
- They should involve different classes/functions with different responsibilities
- They should target different code patterns or architectural concerns
- Avoid picking multiple methods from the same class or methods that do similar things

## Output
After finding all candidates, output a single JSON block:
```json
{{
  "{smell_type}": [
    {{"file": "...", "class_name": "...", "method_name": "...", "line_number": 123, "reason": "..."}}
  ]
}}
```

The key must be exactly: {smell_type}
It must have exactly 5 entries.
"""


def _smell_search_hints(smell_type: str) -> str:
    """Return search hints for finding good injection points."""
    hints = {
        "feature_envy": "Look for methods that already interact with other objects — they have natural cross-object access patterns that make it easy to shift more logic to depend on foreign attributes.",
        "god_classes": "Look for medium-large classes with clear responsibilities — they have enough structure that extra unrelated methods/state can be plausibly added without looking obviously wrong.",
        "data_clumps": "Look for functions with 3+ parameters or methods that share similar parameter groups — easy to duplicate parameter patterns across more call sites.",
        "shotgun_surgery": "Look for well-encapsulated logic in one place — it can be fragmented and scattered across multiple files to create the smell.",
        "dead_code_elimination": "Look for methods with moderate complexity and multiple callers — easy to add plausible-looking but unreachable branches or wrapper functions.",
        "interface_segregation": "Look for abstract base classes or interfaces — easy to bloat them with extra methods that only some implementors need.",
        "deeply_inlined_method": "Look for methods that call helper functions or have moderate nesting — the helpers can be inlined back to create an overly long method.",
    }
    return hints.get(smell_type, "")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_existing_candidates(candidates_path: str) -> Optional[Dict]:
    """Load cached candidates if file exists."""
    if not os.path.exists(candidates_path):
        return None
    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_missing_smell_types(
    existing_data: Optional[Dict],
    smell_types: List[Dict],
) -> List[str]:
    """Return smell type names that don't have 5 candidates yet."""
    if existing_data is None:
        return [s["type"] for s in smell_types]

    candidates = existing_data.get("candidates", {})
    missing = []
    for s in smell_types:
        st = s["type"]
        entries = candidates.get(st, [])
        if len(entries) < 5:
            missing.append(st)
    return missing


def process_repo(
    repo_name: str,
    repo_spec: Dict,
    smell_types: List[Dict],
    project_dir: str,
    output_dir: str,
    force: bool = False,
    timeout: int = 1200,
) -> bool:
    """Find candidates for one repo. Returns True if successful."""
    repo_path = os.path.join(project_dir, repo_name)
    src_path = repo_spec.get("src_path", repo_name)

    if not os.path.isdir(repo_path):
        print(f"  Repo path not found: {repo_path}, skipping")
        return False

    # Check cache
    candidates_path = os.path.join(output_dir, repo_name, "candidates.json")
    existing_data = None if force else load_existing_candidates(candidates_path)

    missing_types = find_missing_smell_types(existing_data, smell_types)
    if not missing_types:
        print(f"  All smell types already have candidates, skipping")
        return True

    print(f"  Missing candidates for: {', '.join(missing_types)}")

    if existing_data is None:
        existing_data = {"repo_name": repo_name, "candidates": {}, "usage": {}}

    # Ensure usage dict exists (for older cached files)
    if "usage" not in existing_data:
        existing_data["usage"] = {}

    os.makedirs(os.path.dirname(candidates_path), exist_ok=True)

    # Collect eligible files
    eligible_files = collect_source_files(repo_path, src_path)
    if not eligible_files:
        print(f"  No eligible files (lines > 500) found in {src_path}")
        return False

    print(f"  Found {len(eligible_files)} eligible files")
    for f in eligible_files[:10]:
        print(f"    {f['file']} ({f['lines']} lines)")
    if len(eligible_files) > 10:
        print(f"    ... and {len(eligible_files) - 10} more")

    # Process each missing smell type one at a time
    smell_type_map = {s["type"]: s for s in smell_types}
    for st in missing_types:
        smell = smell_type_map[st]
        print(f"\n  --- Finding candidates for: {st} ---")

        prompt = build_candidate_prompt(smell, eligible_files, src_path)

        try:
            response_text, trajectory, usage = call_claude_cli(
                prompt, cwd=repo_path, timeout=timeout
            )
        except Exception as e:
            print(f"  Claude CLI call failed for {st}: {e}")
            continue

        parsed = extract_json_from_response(response_text)
        if parsed is None:
            try:
                parsed = json.loads(response_text)
            except (json.JSONDecodeError, TypeError):
                pass

        if parsed is None:
            print(f"  Failed to parse JSON from Claude response for {st}")
            print(f"  Response preview: {response_text[:500]}")
            continue

        entries = parsed.get(st, [])
        if isinstance(entries, list) and entries:
            existing_data["candidates"][st] = entries[:5]
            print(f"  Found {len(entries[:5])} candidates for {st}")
        else:
            print(f"  Warning: no candidates returned for {st}")

        # Record usage for this smell type
        existing_data["usage"][st] = usage

        # Save after each smell type (incremental progress)
        with open(candidates_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cost = usage.get("total_cost_usd", 0.0)
        print(f"  Usage for {st}: in={inp:,} out={out:,} cost=${cost:.4f}")

    print(f"  Saved candidates to {candidates_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find smell injection candidates using Claude Code CLI."
    )
    parser.add_argument(
        "--project-dir", default="../project",
        help="Root directory containing cloned repos.",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--project-name", default=None,
        help="Process a single repo instead of all selected.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-scan even if candidates.json exists.",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout in seconds for Claude CLI call (default: 600).",
    )
    args = parser.parse_args()

    # Load configs
    with open("smell_type.json", "r", encoding="utf-8") as f:
        smell_types = json.load(f)

    with open("repo_list.json", "r", encoding="utf-8") as f:
        repo_list = json.load(f)

    # Filter repos
    selected_repos = {}
    for name, spec in repo_list.items():
        if args.project_name and name != args.project_name:
            continue
        if not args.project_name and not spec.get("selected", False):
            continue
        selected_repos[name] = spec

    if not selected_repos:
        print("No repos selected. Check repo_list.json or --project-name.")
        sys.exit(1)

    print(f"Processing {len(selected_repos)} repo(s): {', '.join(selected_repos.keys())}")
    print(f"Smell types: {', '.join(s['type'] for s in smell_types)}")
    print()

    results = {}
    for repo_name, repo_spec in selected_repos.items():
        print(f"{'='*60}")
        print(f"Repo: {repo_name}")
        print(f"{'='*60}")

        ok = process_repo(
            repo_name=repo_name,
            repo_spec=repo_spec,
            smell_types=smell_types,
            project_dir=args.project_dir,
            output_dir=args.output_dir,
            force=args.force,
            timeout=args.timeout,
        )
        results[repo_name] = "OK" if ok else "FAILED"
        print()

    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
