"""
Smell Benchmark Pipeline
========================
Injects code smells into repositories using an LLM agent, captures diffs,
maps modified functions to unit tests, and produces smell_codes.json.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uuid

from testunits import reset_repository, run_project_tests
from utils import _run_git_command, get_spec, hashcode, prepare_to_run
from find_candidates import process_repo as generate_candidates
from claude_cli import call_claude_cli, extract_json_from_response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FIX_RETRIES = 3
MAX_CANDIDATE_RETRIES = 3
DIFFICULTY_LEVELS = ("easy", "medium")

# Instruction levels for evaluation:
# - "targeted": give smell type + precise location (file, class, method)
# - "guided":   give smell type + related file paths only, agent finds the rest
INSTRUCTION_LEVELS = ("targeted", "guided")

# Difficulty amplifiers appended to the template for hard/expert levels
DIFFICULTY_AMPLIFIERS = {
    "easy": "",
    "medium": "",
    "hard": """### 🔥 Difficulty Amplifiers (REQUIRED)

1. **Camouflage**: The smell must look like an intentional design pattern (e.g., Strategy, Observer, Decorator, Adapter) — not an obvious mistake or code rot.
2. **Diff noise**: Interleave the smell injection with small legitimate improvements (rename a variable for clarity, extract a constant) so the diff is not purely smell-related.
3. **Multi-step resolution**: Fixing the smell must require coordinated changes in 3+ locations — fixing any single location alone should either break tests or leave the smell partially intact.
4. **Indirect coupling**: Use at least one layer of indirection (callbacks, shared mutable state, config-driven dispatch, or dynamic attribute access) to hide the dependency chain.""",
    "expert": """### 🔥 Difficulty Amplifiers (REQUIRED — Expert Level)

1. **Camouflage**: The smell MUST masquerade as a recognized design pattern (e.g., Strategy, Mediator, Template Method, Abstract Factory). An experienced developer reviewing the code should initially believe the pattern is intentional.
2. **Red herrings**: Introduce 1-2 code regions that superficially resemble the same smell type but are actually correct / well-designed. The agent must distinguish the real smell from the decoys.
3. **Semantic dependency**: The smell should only be identifiable by understanding the domain/business logic — pure structural or syntactic analysis must be insufficient. For example, two functions that look independent but operate on the same conceptual entity.
4. **Entanglement**: Deeply interleave smell-related changes with legitimate code so that a naive "revert the diff" approach would break functionality. Add small behavioral improvements that must be preserved.
5. **Multi-step resolution**: Fixing the smell must require coordinated, order-dependent changes in 4+ locations. Partial fixes must leave the code in a worse state than the smell itself.
6. **Indirect coupling**: Use at least two layers of indirection (e.g., registry + callback, config + dynamic import, decorator + shared state) to hide the real dependency chain from static analysis.""",
}


def print_usage_summary(usage_records: List[Dict]):
    """Print a table summarising token usage across all invocations."""
    print(f"\n{'='*60}")
    print("Token Usage Summary")
    print(f"{'='*60}")
    print(f"  {'Smell Type':<30} {'In Tokens':>12} {'Out Tokens':>12} {'Cost(USD)':>10} {'Time(s)':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    total_in = total_out = 0
    total_cost = 0.0
    total_dur = 0

    for rec in usage_records:
        name = rec.get("smell_type", "?")[:30]
        u = rec.get("usage", {})
        inp = u.get("input_tokens", 0)
        out = u.get("output_tokens", 0)
        cost = u.get("total_cost_usd", 0.0)
        dur = u.get("duration_ms", 0)
        total_in += inp
        total_out += out
        total_cost += cost
        total_dur += dur
        print(f"  {name:<30} {inp:>12,} {out:>12,} ${cost:>9.4f} {dur/1000:>7.1f}")

    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
    print(f"  {'Total':<30} {total_in:>12,} {total_out:>12,} ${total_cost:>9.4f} {total_dur/1000:>7.1f}")


# ---------------------------------------------------------------------------
# Test execution helpers
# ---------------------------------------------------------------------------

def apply_diff_and_test(
    repo_name: str,
    repo_path: str,
    smell_content: str,
    testsuites: List[str],
    commit_id: str,
    test_cmd: str = "",
    envs: Optional[Dict] = None,
) -> Tuple[bool, str]:
    """Apply a smell diff to the repo, run mapped tests, then reset.

    Returns (passed, test_output).
    """
    envs = envs or {}
    reset_repository(repo_path, commit_id)

    # Apply the diff
    uuid_str = str(uuid.uuid4())
    diff_file = os.path.join(repo_path, f"{repo_name}_smell_{uuid_str}.diff")
    try:
        with open(diff_file, "w") as f:
            f.write(smell_content)
        _run_git_command(["apply", os.path.abspath(diff_file)], cwd=repo_path)
    except Exception as e:
        reset_repository(repo_path, commit_id)
        return False, f"Failed to apply diff: {e}"
    finally:
        if os.path.exists(diff_file):
            os.remove(diff_file)

    # Run tests
    try:
        success, output = run_project_tests(
            repo_name, repo_path, testsuites,
            envs=envs, test_cmd=test_cmd, timeout=60,
        )
        return bool(success), output or ""
    except Exception as e:
        return False, f"Test execution error: {e}"
    finally:
        reset_repository(repo_path, commit_id)


def build_fix_prompt(
    smell_type: str,
    smell_content: str,
    test_scripts: List[str],
    test_error_output: str,
) -> str:
    """Build a prompt asking the agent to fix test failures after smell injection."""
    tests_str = "\n".join(f"- {t}" for t in test_scripts)
    # Truncate very long test output
    if len(test_error_output) > 4000:
        test_error_output = test_error_output[:4000] + "\n... (truncated)"

    return f"""You previously injected a "{smell_type}" code smell into this codebase.
The changes you made caused the following unit tests to FAIL.

Your task: fix the code so that all tests pass while KEEPING the injected code smell intact.
The smell must still be present and non-trivial after your fix.

Do NOT run any tests yourself — testing is handled externally.

## Diff of your previous changes
```diff
{smell_content}
```

## Failing test scripts
{tests_str}

## Test error output
```
{test_error_output}
```

## Requirements
1. Fix the failing tests while preserving the code smell injection
2. The code must compile/run correctly
3. Do NOT remove the smell — only fix the breakage
4. Do NOT create new files
5. Do NOT run any test commands (pytest, unittest, etc.)

After making your fixes, output the same JSON format as before:
```json
{{
  "hint_targeted": "Natural language task: tell agent to find and refactor the smell. Include smell type + file + class/method. No fixed format.",
  "hint_guided": "Natural language task: tell agent to find and refactor the smell. Include smell type + primary class/method + related files. No fixed format.",
  "smell_function": ["<absolute_file_path>", "<class name or null>", "<function name or null>"],
  "test_functions": [["<absolute_file_path>", "<class name or null>", "<function_name>"]]
}}
```
"""


def save_attempt(
    attempt_dir: str,
    trajectory: List[Dict],
    smell_content: str,
    test_output: str,
    test_passed: bool,
    usage: Dict,
    parsed_json: Optional[Dict] = None,
):
    """Persist all artifacts for one attempt into the given directory."""
    os.makedirs(attempt_dir, exist_ok=True)

    with open(os.path.join(attempt_dir, "trajectory.json"), "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    with open(os.path.join(attempt_dir, "smell.diff"), "w", encoding="utf-8") as f:
        f.write(smell_content)

    with open(os.path.join(attempt_dir, "test_output.txt"), "w", encoding="utf-8") as f:
        f.write(test_output)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_passed": test_passed,
        "usage": usage,
        "event_count": len(trajectory),
    }
    if parsed_json is not None:
        meta["parsed_json"] = parsed_json
    with open(os.path.join(attempt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def load_template(template_path: str = "template/smell_template.md") -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(
    template: str,
    smell_type: str,
    smell_desc: str,
    project_path: str,
    difficulty: str = "easy",
    smell_config: Optional[Dict] = None,
    target_file: str = "",
    target_file_lines: int = 0,
    candidate: Optional[Dict] = None,
) -> str:
    prompt = template.replace("[SMELL_TYPE]", smell_type)
    prompt = prompt.replace("[PROJECT_PATH]", project_path)
    prompt = prompt.replace("[SMELL_TYPE_DESC]", smell_desc)

    # Difficulty-level substitutions
    diff_levels = (smell_config or {}).get("difficulty_levels", {})
    diff_info = diff_levels.get(difficulty, {})
    diff_strategy = diff_info.get("strategy", "")
    min_files = diff_info.get("min_files", 2)

    prompt = prompt.replace("[DIFFICULTY]", difficulty)
    prompt = prompt.replace("[DIFFICULTY_DESC]", diff_strategy)
    prompt = prompt.replace("[MIN_FILES]", str(min_files))
    prompt = prompt.replace("[DIFFICULTY_AMPLIFIERS]", DIFFICULTY_AMPLIFIERS.get(difficulty, ""))

    # Target candidate substitutions
    prompt = prompt.replace("[TARGET_FILE]", target_file or "(not specified)")
    prompt = prompt.replace("[TARGET_FILE_LINES]", str(target_file_lines) if target_file_lines else "?")

    if candidate:
        cls = candidate.get("class_name") or ""
        method = candidate.get("method_name") or ""
        if cls and method:
            class_method = f"{cls}.{method}"
        elif cls:
            class_method = cls
        else:
            class_method = method or "(not specified)"
        line_num = str(candidate.get("line_number", "?"))
    else:
        class_method = "(not specified)"
        line_num = "?"

    prompt = prompt.replace("[TARGET_CLASS_METHOD]", class_method)
    prompt = prompt.replace("[TARGET_LINE_NUMBER]", line_num)

    return prompt


# ---------------------------------------------------------------------------
# Function -> test mapping
# ---------------------------------------------------------------------------

def load_function_test_mapping(mapping_path: str) -> Dict:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_path_to_module(file_path: str, src_path: str, repo_path: str) -> str:
    """Convert an absolute or relative file path to a dotted module path.

    E.g. /repo/click/src/click/core.py with src_path=src/click -> click.core
    """
    # Normalise to relative path within repo
    if os.path.isabs(file_path):
        rel = os.path.relpath(file_path, repo_path)
    else:
        rel = file_path

    # Strip src_path prefix if present
    src_parts = Path(src_path).parts
    rel_parts = Path(rel).parts

    # Try to match the src_path prefix
    if len(rel_parts) >= len(src_parts):
        if rel_parts[: len(src_parts)] == src_parts:
            rel_parts = rel_parts[len(src_parts):]
        else:
            # The file might already be relative to src_path
            pass

    # Build module path: remove .py, join with dots
    # The first component of src_path's last part is the package name
    package_name = Path(src_path).parts[-1]
    module_parts = list(rel_parts)
    if module_parts and module_parts[-1].endswith(".py"):
        module_parts[-1] = module_parts[-1][:-3]
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]

    return package_name + "." + ".".join(module_parts) if module_parts else package_name


def build_function_key(module_path: str, class_name: Optional[str], func_name: str) -> str:
    """Build the lookup key as used in function_testunit_mapping.json.

    Format: module.path:ClassName.method_name  or  module.path:function_name
    """
    if class_name:
        return f"{module_path}:{class_name}.{func_name}"
    return f"{module_path}:{func_name}"


def find_tests_for_functions(
    main_functions: List,
    mapping: Dict,
    src_path: str,
    repo_path: str,
    max_tests_per_func: int = 10,
) -> List[str]:
    """Look up tests for each test_functions entry. Returns deduplicated list of test paths."""
    functions_map = mapping.get("functions", {})
    all_tests = []
    seen = set()

    for func_entry in main_functions:
        if not isinstance(func_entry, list) or len(func_entry) < 3:
            continue
        file_path, class_name, func_name = func_entry[0], func_entry[1], func_entry[2]

        # Build the key
        module_path = file_path_to_module(file_path, src_path, repo_path)
        key = build_function_key(module_path, class_name, func_name)

        # Look up in mapping
        func_info = functions_map.get(key)
        if func_info is None:
            # Try without class (in case class_name is wrong or None mismatch)
            alt_key = build_function_key(module_path, None, func_name)
            func_info = functions_map.get(alt_key)
        if func_info is None:
            # Fuzzy match: search for keys ending with the function name
            for k, v in functions_map.items():
                if k.endswith(f".{func_name}") or k.endswith(f":{func_name}"):
                    func_info = v
                    break

        if func_info is None:
            print(f"  Warning: no test mapping found for {key}")
            continue

        tests = func_info.get("tests", [])
        if len(tests) > max_tests_per_func:
            tests = random.sample(tests, max_tests_per_func)

        for t in tests:
            if t not in seen:
                seen.add(t)
                all_tests.append(t)

    return all_tests


# ---------------------------------------------------------------------------
# Diff capture (following base_method.py create_diff_file pattern)
# ---------------------------------------------------------------------------

def capture_diffs(repo_path: str, commit_id: str) -> Tuple[str, str]:
    """Capture smell_content and gt_content diffs.

    Assumes the agent has already modified files in the repo.

    Returns:
        (smell_content, gt_content)
    """
    # smell_content: diff from original commit to current (modified) state
    smell_ret = _run_git_command(
        ["diff", "--ignore-space-at-eol", commit_id], cwd=repo_path
    )
    smell_content = smell_ret.stdout

    if not smell_content.strip():
        return "", ""

    # Stage and commit the smell changes
    _run_git_command(["add", "-A"], cwd=repo_path)

    # Check if there are staged changes
    staged_check = _run_git_command(["diff", "--cached", "--quiet"], check=False, cwd=repo_path)
    if staged_check.returncode == 0:
        # No staged changes
        return smell_content, ""

    _run_git_command(["commit", "-m", "[baseline] smell"], cwd=repo_path)
    smell_commit = _run_git_command(["rev-parse", "HEAD"], cwd=repo_path).stdout.strip()

    # Restore original files by resetting file contents to commit_id
    _run_git_command(["checkout", commit_id, "--", "."], cwd=repo_path)

    # gt_content: diff from smell commit to restored original
    gt_ret = _run_git_command(
        ["diff", "--ignore-space-at-eol", smell_commit], cwd=repo_path
    )
    gt_content = gt_ret.stdout

    return smell_content, gt_content


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_one_smell(
    template: str,
    smell_type: str,
    smell_desc: str,
    repo_name: str,
    repo_spec: Dict,
    repo_path: str,
    mapping: Dict,
    output_dir: str,
    difficulty: str = "easy",
    smell_config: Optional[Dict] = None,
    target_file: str = "",
    target_file_lines: int = 0,
    assignment_key: str = "",
    candidate: Optional[Dict] = None,
) -> Optional[Dict]:
    """Process a single (repo, smell_type) combination.

    Flow:
    1. Call agent to inject smell
    2. Capture diff, parse JSON, find mapped tests
    3. Run only the mapped tests
    4. If tests fail: save attempt, send errors back to agent to fix, repeat
    5. On success: return result dict
    """
    commit_id = repo_spec["commit_id"]
    src_path = repo_spec.get("src_path", repo_name)
    smell_type_slug = smell_type.replace(" ", "_")

    # Per-smell output directory: {output_dir}/{smell_type_slug}/{difficulty}/
    smell_dir = os.path.join(output_dir, smell_type_slug, difficulty)
    os.makedirs(smell_dir, exist_ok=True)

    # --- Attempt 0: initial agent call ---
    reset_repository(repo_path, commit_id)
    prompt = render_prompt(template, smell_type, smell_desc, src_path, difficulty, smell_config, target_file, target_file_lines, candidate)

    print(f"  Calling claude CLI for {repo_name} / {smell_type} ...")
    try:
        response_text, trajectory, usage = call_claude_cli(prompt, cwd=repo_path)
    except Exception as e:
        print(f"  Claude CLI call failed: {e}")
        reset_repository(repo_path, commit_id)
        return None

    # Capture diffs before reset
    smell_content, gt_content = capture_diffs(repo_path, commit_id)
    if not smell_content.strip():
        print(f"  No diff produced for {repo_name} / {smell_type}")
        reset_repository(repo_path, commit_id)
        return None

    # Parse JSON from response
    parsed = extract_json_from_response(response_text)
    if parsed is None:
        print(f"  Failed to parse JSON from agent response for {repo_name} / {smell_type}")
        reset_repository(repo_path, commit_id)
        return None

    hint_targeted = parsed.get("hint_targeted", "")
    hint_guided = parsed.get("hint_guided", "")
    smell_function = parsed.get("smell_function", [])
    test_functions = parsed.get("test_functions", parsed.get("main_function", []))
    if test_functions and not isinstance(test_functions[0], list):
        test_functions = [test_functions]

    # Find mapped tests
    testsuites = find_tests_for_functions(
        test_functions, mapping, src_path, repo_path
    )
    if not testsuites:
        print(f"  No tests found for modified functions in {repo_name} / {smell_type}")
        reset_repository(repo_path, commit_id)
        return None

    # --- Test & fix loop ---
    total_usage = dict(usage)  # accumulate across retries
    for attempt in range(MAX_FIX_RETRIES + 1):
        attempt_label = f"attempt_{attempt}"
        attempt_dir = os.path.join(smell_dir, attempt_label)

        print(f"  [{attempt_label}] Running {len(testsuites)} mapped tests ...")
        test_passed, test_output = apply_diff_and_test(
            repo_name=repo_name,
            repo_path=repo_path,
            smell_content=smell_content,
            testsuites=testsuites,
            commit_id=commit_id,
            test_cmd=repo_spec.get("test_cmd", ""),
            envs=repo_spec.get("envs", {}),
        )

        # Save this attempt
        save_attempt(
            attempt_dir=attempt_dir,
            trajectory=trajectory,
            smell_content=smell_content,
            test_output=test_output,
            test_passed=test_passed,
            usage=usage,
            parsed_json=parsed,
        )

        if test_passed:
            print(f"  [{attempt_label}] Tests PASSED for {repo_name} / {smell_type}")
            break

        print(f"  [{attempt_label}] Tests FAILED for {repo_name} / {smell_type}")

        # No more retries left
        if attempt >= MAX_FIX_RETRIES:
            print(f"  Exhausted {MAX_FIX_RETRIES} fix retries, giving up on {smell_type}")
            reset_repository(repo_path, commit_id)
            return None

        # Ask agent to fix
        print(f"  [{attempt_label}] Sending test errors to agent for fix ...")
        reset_repository(repo_path, commit_id)
        fix_prompt = build_fix_prompt(
            smell_type=smell_type,
            smell_content=smell_content,
            test_scripts=testsuites,
            test_error_output=test_output,
        )

        try:
            response_text, trajectory, usage = call_claude_cli(fix_prompt, cwd=repo_path)
        except Exception as e:
            print(f"  Fix call failed: {e}")
            reset_repository(repo_path, commit_id)
            return None

        # Accumulate usage
        for k in ("input_tokens", "output_tokens", "cache_creation_tokens",
                   "cache_read_tokens", "duration_ms"):
            total_usage[k] = total_usage.get(k, 0) + usage.get(k, 0)
        total_usage["total_cost_usd"] = total_usage.get("total_cost_usd", 0.0) + usage.get("total_cost_usd", 0.0)

        # Re-capture diff after fix
        new_smell_content, new_gt_content = capture_diffs(repo_path, commit_id)
        if not new_smell_content.strip():
            print(f"  Fix produced no diff, retrying ...")
            continue
        smell_content = new_smell_content
        gt_content = new_gt_content

        # Re-parse JSON (agent may have updated it)
        new_parsed = extract_json_from_response(response_text)
        if new_parsed:
            parsed = new_parsed
            hint_targeted = parsed.get("hint_targeted", hint_targeted)
            hint_guided = parsed.get("hint_guided", hint_guided)
            new_sf = parsed.get("smell_function", [])
            if new_sf:
                smell_function = new_sf
            new_tf = parsed.get("test_functions", parsed.get("main_function", []))
            if new_tf:
                if not isinstance(new_tf[0], list):
                    new_tf = [new_tf]
                test_functions = new_tf

    else:
        # Loop finished without break => all retries exhausted
        reset_repository(repo_path, commit_id)
        return None

    # --- Build final result ---
    h = hashcode(smell_content)
    instance_id = f"{repo_name}-{smell_type_slug}-{h}"

    # Normalize file paths to be relative to project/{repo_name}
    def _normalize_funcs(funcs):
        normalized = []
        for func_entry in funcs:
            entry = list(func_entry)
            fp = entry[0]
            if os.path.isabs(fp):
                try:
                    entry[0] = os.path.relpath(fp, repo_path)
                except ValueError:
                    pass
            normalized.append(entry)
        return normalized

    normalized_smell_function = list(smell_function) if smell_function else []
    if normalized_smell_function and os.path.isabs(normalized_smell_function[0]):
        try:
            normalized_smell_function[0] = os.path.relpath(normalized_smell_function[0], repo_path)
        except ValueError:
            pass
    normalized_test_functions = _normalize_funcs(test_functions)

    reset_repository(repo_path, commit_id)

    result = {
        "instance_id": instance_id,
        "type": smell_type,
        "difficulty": difficulty,
        "target_file": target_file,
        "assignment_key": assignment_key,
        "hint_targeted": hint_targeted,
        "hint_guided": hint_guided,
        "smell_function": normalized_smell_function,
        "test_functions": normalized_test_functions,
        "testsuites": testsuites,
        "smell_content": smell_content,
        "gt_content": gt_content,
        "hash": h,
        "commit_hash": commit_id,
        "project_name": repo_name,
        "usage": total_usage,
    }

    # Write this case immediately to its own directory
    result_path = os.path.join(smell_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Written case to {result_path}")

    return result


def main(args):
    random.seed(args.seed)

    # Load config files
    with open("smell_type.json", "r", encoding="utf-8") as f:
        smell_types = json.load(f)

    with open("repo_list.json", "r", encoding="utf-8") as f:
        repo_list = json.load(f)

    template = load_template()

    # Filter to selected repos (or single repo if specified)
    selected_repos = {}
    for name, spec in repo_list.items():
        if args.project_name and name != args.project_name:
            continue
        if not args.project_name and not spec.get("selected", False):
            continue
        selected_repos[name] = spec

    if not selected_repos:
        print("No repos selected. Check repo_list.json or --project-name.")
        return

    project_dir = args.project_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Load existing smell_codes.json for skip logic ---
    smell_codes_path = os.path.join(output_dir, "smell_codes.json")
    existing_entries: List[Dict] = []
    if not args.force and os.path.exists(smell_codes_path):
        try:
            with open(smell_codes_path, "r", encoding="utf-8") as f:
                existing_entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_entries = []

    # Build set of completed (repo, smell_type, difficulty) triples
    completed_triples = set()
    for entry in existing_entries:
        key = (entry.get("project_name", ""),
               entry.get("type", ""),
               entry.get("difficulty", ""))
        completed_triples.add(key)

    if completed_triples:
        print(f"Loaded {len(completed_triples)} completed cases from {smell_codes_path}")

    all_results = list(existing_entries)  # start from existing data

    for repo_name, repo_spec in selected_repos.items():
        repo_spec["name"] = repo_name
        repo_path = os.path.join(project_dir, repo_name)
        repo_output_dir = os.path.join(output_dir, repo_name)
        os.makedirs(repo_output_dir, exist_ok=True)

        # Prepare environment
        spec = get_spec(repo_name)
        if spec is None:
            print(f"Spec not found for {repo_name}, skipping")
            continue
        if not prepare_to_run(spec, project_path=project_dir):
            print(f"Failed to prepare environment for {repo_name}, skipping")
            continue

        # Load function-test mapping
        mapping_path = os.path.join(repo_output_dir, "function_testunit_mapping.json")
        if not os.path.exists(mapping_path):
            print(f"function_testunit_mapping.json not found for {repo_name}, skipping")
            continue
        mapping = load_function_test_mapping(mapping_path)

        commit_id = repo_spec["commit_id"]

        usage_records = []
        print(f"\n{'='*60}")
        print(f"Processing repo: {repo_name}")
        print(f"{'='*60}")

        # Load candidates (pre-computed by find_candidates.py), generate if missing
        candidates_path = os.path.join(repo_output_dir, "candidates.json")
        candidates_data = {}
        if os.path.exists(candidates_path):
            try:
                with open(candidates_path, "r", encoding="utf-8") as f:
                    candidates_data = json.load(f).get("candidates", {})
            except (json.JSONDecodeError, OSError):
                candidates_data = {}

        # Check if any smell types are missing candidates
        missing_candidate_types = [
            s["type"] for s in smell_types
            if not candidates_data.get(s["type"])
        ]
        if missing_candidate_types:
            print(f"  Missing candidates for {len(missing_candidate_types)} smell types, generating...")
            generate_candidates(
                repo_name=repo_name,
                repo_spec=repo_spec,
                smell_types=smell_types,
                project_dir=project_dir,
                output_dir=output_dir,
            )
            # Reload after generation
            if os.path.exists(candidates_path):
                try:
                    with open(candidates_path, "r", encoding="utf-8") as f:
                        candidates_data = json.load(f).get("candidates", {})
                    print(f"  Loaded candidates from {candidates_path}")
                except (json.JSONDecodeError, OSError):
                    print(f"  Warning: failed to load candidates after generation")
        else:
            print(f"  Loaded candidates from {candidates_path}")

        # Build list of (smell, difficulty) pairs that still need processing
        # Note: instruction_level (targeted/guided) is an evaluation-time concern.
        # The injection pipeline produces one smell per (smell_type, difficulty)
        # and outputs both hint_targeted and hint_guided for later use.
        pending_tasks = []
        for smell in smell_types:
            for difficulty in DIFFICULTY_LEVELS:
                smell_type = smell["type"]
                if (repo_name, smell_type, difficulty) in completed_triples:
                    continue
                pending_tasks.append((smell, difficulty))

        total_tasks = len(smell_types) * len(DIFFICULTY_LEVELS)
        completed_count = total_tasks - len(pending_tasks)
        print(f"  {len(pending_tasks)} tasks pending "
              f"({completed_count} already completed, "
              f"{len(smell_types)} smell types x {len(DIFFICULTY_LEVELS)} levels)")

        if not pending_tasks:
            print(f"  All smell type x difficulty combinations already completed for {repo_name}, skipping")
            continue

        for task_idx, (smell, difficulty) in enumerate(pending_tasks):
            smell_type = smell["type"]
            smell_desc = smell.get("desc", "")

            # Pick a random candidate for this smell type, retry with different candidates on failure
            smell_candidates = candidates_data.get(smell_type, [])
            if not smell_candidates:
                print(f"  No candidates available for {smell_type}, skipping")
                continue

            tried_indices = set()
            result = None
            for candidate_attempt in range(min(MAX_CANDIDATE_RETRIES, len(smell_candidates))):
                # Pick a candidate we haven't tried yet
                remaining = [i for i in range(len(smell_candidates)) if i not in tried_indices]
                if not remaining:
                    break
                chosen_idx = random.choice(remaining)
                tried_indices.add(chosen_idx)
                candidate = smell_candidates[chosen_idx]

                target_file = candidate.get("file", "")
                cls = candidate.get("class_name") or ""
                method = candidate.get("method_name") or ""
                label = f"{cls}.{method}" if cls and method else (cls or method)
                akey = f"{smell_type}::{difficulty}::{target_file}::{label}"

                if candidate_attempt == 0:
                    print(f"\n--- [{task_idx+1}/{len(pending_tasks)}] "
                          f"{smell_type} ({difficulty}) -> {target_file} : {label} ---")
                else:
                    print(f"  Retry {candidate_attempt}/{MAX_CANDIDATE_RETRIES} with different candidate: "
                          f"{target_file} : {label}")

                reset_repository(repo_path, commit_id)
                result = process_one_smell(
                    template=template,
                    smell_type=smell_type,
                    smell_desc=smell_desc,
                    repo_name=repo_name,
                    repo_spec=repo_spec,
                    repo_path=repo_path,
                    mapping=mapping,
                    output_dir=repo_output_dir,
                    difficulty=difficulty,
                    smell_config=smell,
                    target_file=target_file,
                    assignment_key=akey,
                    candidate=candidate,
                )

                if result:
                    break
                print(f"  Candidate failed for {smell_type} ({difficulty}), "
                      f"attempt {candidate_attempt+1}/{min(MAX_CANDIDATE_RETRIES, len(smell_candidates))}")

            if result:
                all_results.append(result)
                usage_records.append({
                    "smell_type": smell_type,
                    "difficulty": difficulty,
                    "usage": result.get("usage", {}),
                })
                print(f"  Success: {result['instance_id']} ({len(result['testsuites'])} tests)")

                # Append to global smell_codes.json immediately
                with open(smell_codes_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
            else:
                print(f"  Skipped: no valid result for {smell_type} ({difficulty}) after {len(tried_indices)} candidate(s)")

        # Print usage summary for this repo
        if usage_records:
            print_usage_summary(usage_records)

        print(f"\nFinished repo {repo_name}")

    print(f"\nDone. Total {len(all_results)} cases in {smell_codes_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code smell injection benchmark pipeline.")
    parser.add_argument("--project-dir", default="../project", help="Root directory containing cloned repos.")
    parser.add_argument("--output-dir", default="output", help="Output directory for results.")
    parser.add_argument("--project-name", default=None, help="Process a single repo instead of all selected.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
