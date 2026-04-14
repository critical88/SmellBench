"""Extract smell_content and gt_content from smell_codes.json and apply them to the repo.

Scans all smell_codes.json files under tmp_code_benchmark/output/<repo_name>/,
and for each refactor_code entry:
1. git reset --hard to the specified commit_id
2. Apply smell_content (introduces the code smell)
3. Commit the smell code
4. Apply gt_content (applies the ground truth fix)
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from testunits import run_project_tests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR =  BASE_DIR / "output"
PROJECT_DIR = BASE_DIR.parent / "project"


def run_git(repo_dir: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )


def apply_diff(repo_dir: str, diff_content: str, label: str) -> bool:
    """Write diff to a temp file and apply it via git apply."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_content)
        patch_path = f.name

    result = run_git(repo_dir, "apply", "--allow-empty", patch_path)
    Path(patch_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"    [ERROR] Failed to apply {label}:")
        print(f"      stderr: {result.stderr.strip()}")
        return False

    print(f"    [OK] Applied {label}")
    return True


def process_smell_codes(json_path: Path, repo_dir: str):
    with open(json_path) as f:
        data = json.load(f)

    settings = data["settings"]
    repo_name = data["name"]
    commit_id = settings["commit_id"]
    test_cmd = settings.get("test_cmd", "")
    envs = settings.get("envs", {})
    refactor_codes = data["refactor_codes"]

    if not refactor_codes:
        print(f"  No refactor_codes in {json_path}")
        return

    print(f"  Base commit: {commit_id}")
    print(f"  Entries: {len(refactor_codes)}")

    for i, entry in enumerate(refactor_codes):
        iid = entry["instance_id"]
        entry_commit = entry.get("commit_hash", commit_id)

        print(f"  [{i + 1}/{len(refactor_codes)}] {iid}")

        # Step 1: reset to commit
        result = run_git(repo_dir, "reset", "--hard", entry_commit)
        if result.returncode != 0:
            print(f"    [ERROR] git reset --hard {entry_commit} failed: {result.stderr.strip()}")
            continue
        print(f"    [OK] Reset to {entry_commit[:12]}")

        # Step 2: apply smell_content
        smell_ok = apply_diff(repo_dir, entry["smell_content"], "smell_content")
        if not smell_ok:
            continue

        # Step 3: run testsuites to verify smell_content preserves behavior
        testsuites = entry.get("testsuites", [])
        if testsuites:
            passed, output = run_project_tests(
                repo_name, repo_dir, testsuites,
                envs=envs, test_cmd=test_cmd, timeout=120,
            )
            if passed:
                print(f"    [OK] Tests passed after smell_content")
            else:
                print(f"    [FAIL] Tests failed after smell_content, skipping entry")
                print(f"      {output[:500] if output else ''}")
                continue
        else:
            print(f"    [WARN] No testsuites defined, skipping test verification")

        # Step 4: commit the smell code
        run_git(repo_dir, "add", "-A")
        result = run_git(repo_dir, "commit", "-m", f"Apply smell: {iid}")
        if result.returncode != 0:
            print(f"    [ERROR] Failed to commit smell_content: {result.stderr.strip()}")
            continue
        print(f"    [OK] Committed smell_content")

        # Step 5: apply gt_content
        apply_diff(repo_dir, entry["gt_content"], "gt_content")



def main():
    parser = argparse.ArgumentParser(
        description="Scan all smell_codes.json under a repo_name and apply diffs to the repo."
    )
    parser.add_argument(
        "repo_name",
        help="Name of the repo (e.g. click, flask). Scans tmp_code_benchmark/output/<repo_name>/ for smell_codes.json files.",
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="Path to the git repo. Defaults to project/<repo_name>.",
    )
    args = parser.parse_args()

    repo_name = args.repo_name
    repo_output_dir = OUTPUT_DIR / repo_name

    if not repo_output_dir.is_dir():
        print(f"Output directory not found: {repo_output_dir}")
        sys.exit(1)

    repo_dir = args.repo_dir or str(PROJECT_DIR / repo_name)
    if not Path(repo_dir).is_dir():
        print(f"Repo directory not found: {repo_dir}")
        sys.exit(1)

    # Find all smell_codes.json files under output/<repo_name>/
    json_files = sorted(repo_output_dir.rglob("smell_codes.json"))

    if not json_files:
        print(f"No smell_codes.json found under {repo_output_dir}")
        sys.exit(1)

    print(f"Repo: {repo_dir}")
    print(f"Found {len(json_files)} smell_codes.json file(s) under {repo_output_dir}")
    print()

    for json_path in json_files:
        rel_path = json_path.relative_to(repo_output_dir)
        print(f"=== Processing {rel_path} ===")
        process_smell_codes(json_path, repo_dir)
        print()


if __name__ == "__main__":
    main()
