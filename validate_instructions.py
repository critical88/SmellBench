"""
Validate and regenerate hint_targeted / hint_guided instructions in code_smells.json.

For each entry, checks whether the instruction fields conform to the rules:
  - hint_targeted (or "easy"): must mention smell type + specific file path +
    class/method name.
  - hint_guided (or "hard"): must mention smell type + related file paths,
    but must NOT reveal class/method names or line numbers.

Entries that fail validation are sent to an LLM for regeneration.
The code_smells.json is updated in place.

Usage:
    python validate_instructions.py --repo-name click
    python validate_instructions.py --repo-name click --dry-run
    python validate_instructions.py --repo-name click --force
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

from claude_cli import call_llm


# ---------------------------------------------------------------------------
# Field name helpers — older entries use "easy"/"hard", newer use
# "hint_targeted"/"hint_guided"
# ---------------------------------------------------------------------------

def _get_targeted(entry: Dict) -> str:
    return entry.get("hint_targeted") or entry.get("easy") or ""


def _get_guided(entry: Dict) -> str:
    return entry.get("hint_guided") or entry.get("hard") or ""


def _get_open(entry: Dict) -> str:
    return entry.get("hint_open") or ""


def _set_targeted(entry: Dict, value: str) -> None:
    if "hint_targeted" in entry:
        entry["hint_targeted"] = value
    else:
        entry["easy"] = value


def _set_guided(entry: Dict, value: str) -> None:
    if "hint_guided" in entry:
        entry["hint_guided"] = value
    else:
        entry["hard"] = value


def _set_open(entry: Dict, value: str) -> None:
    entry["hint_open"] = value


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------



VALIDATE_AND_REGENERATE_PROMPT = """You are a QA reviewer for a code-smell benchmark. Validate the existing instructions and regenerate any that have issues.

**CRITICAL**: Be LENIENT, not overly strict. Only reject if the hint clearly violates the rules below. Focus on checking if instructions LEAK SPECIFIC INFORMATION about what the problem is, not on stylistic preferences or generic quality terms.

When in doubt, ACCEPT the hint rather than reject it.

## Smell Info
- **Smell type**: {smell_type}
- **Smell description**: {smell_description}
- **Smell function**: file=`{smell_file}`, class=`{smell_class}`, method=`{smell_method}`
- **Files changed**: {changed_files}

## Diff
```diff
{smell_diff}
```

## Current Instructions

**hint_targeted**: {hint_targeted}
**hint_guided**: {hint_guided}
**hint_open**: {hint_open}

## Validation Rules

**hint_targeted**:
- Should be a concise, professional task description (1-2 sentences).
- MUST specify the smell type and exact location (file, class, method).
- REJECT ONLY if it:
  * Describes implementation details (how the code works internally)
  * Describes mechanisms (the technical approach or pattern used)
  * Describes architecture (system design or relationships between components)
  * Describes what the code does (its functionality or purpose)
  * Reveals multiple affected files (should focus on the main location only)

**hint_guided**:
- Should be a concise, professional task description (1-2 sentences).
- MUST specify ONLY the smell type and the single main file path.
- REJECT ONLY if it:
  * Mentions other files beyond the main file
  * Mentions class names or method names (agent should discover these)
  * Mentions line numbers
  * Describes implementation details
  * Describes how the smell manifests (the specific pattern or structure)

**hint_open**:
- Should be a concise, professional task description (1-2 sentences).
- MUST be COMPLETELY GENERIC - only file path(s) and a generic refactoring request.
- Should clearly indicate it's a refactoring task, not a vague review or check.
- **ALLOWED generic quality terms** (DO NOT REJECT these): maintainability, maintainable, readability, readable, code quality, better practices, cleaner code, etc.
- REJECT ONLY if it:
  * Hints at the SPECIFIC problem type using words like: design issues, responsibilities, encapsulation, coupling, cohesion, complexity, duplication, dependencies, abstraction, inheritance, modularity, separation of concerns, god class, feature envy, dead code, etc.
  * Reveals the smell type explicitly
  * Mentions specific class names, method names, or function names
  * Mentions line numbers
  * Describes what specific issues exist or what needs to be fixed

**IMPORTANT**: The key is to reject hints that reveal WHAT the problem is, not generic quality improvement language. Generic terms like "maintainability" or "readability" are acceptable, but specific problem indicators like "coupling" or "god class" are not.

## Your Task

For each hint:
1. Check if it violates the rules above
2. If OK, return true and leave the new field empty
3. If problematic, return false, explain the issue, and generate a new hint that fixes it

When regenerating, use varied phrasing and different verbs to increase diversity.

## Output Format

<targeted_ok>true or false</targeted_ok>
<targeted_reason>If false, explain what rule was violated. If true, write "none".</targeted_reason>
<targeted_new>New hint if false, otherwise leave empty</targeted_new>

<guided_ok>true or false</guided_ok>
<guided_reason>If false, explain what rule was violated. If true, write "none".</guided_reason>
<guided_new>New hint if false, otherwise leave empty</guided_new>

<open_ok>true or false</open_ok>
<open_reason>If false, explain what rule was violated. If true, write "none".</open_reason>
<open_new>New hint if false, otherwise leave empty</open_new>
"""


def _parse_xml_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_changed_files(diff_text: str) -> List[str]:
    """Extract file paths from a unified diff."""
    files = []
    for m in re.finditer(r"^diff --git a/(.+?) b/", diff_text, re.MULTILINE):
        f = m.group(1)
        if f not in files:
            files.append(f)
    return files


def validate_and_regenerate_entry(
    entry: Dict,
    smell_types: Dict[str, Dict],
    model: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and regenerate instructions in a single LLM call.

    Returns dict with: targeted_ok, targeted_reason, targeted_new, guided_ok, guided_reason, guided_new, open_ok, open_reason, open_new, usage
    """
    smell_type = entry.get("type", "")
    smell_desc = smell_types.get(smell_type, {}).get("desc", "")
    smell_function = entry.get("smell_function", [])
    changed_files = _extract_changed_files(entry.get("smell_content", ""))

    smell_file = smell_function[0] if len(smell_function) > 0 else ""
    smell_class = smell_function[1] if len(smell_function) > 1 else "null"
    smell_method = smell_function[2] if len(smell_function) > 2 else "null"

    # Truncate very long diffs
    diff_text = entry.get("smell_content", "")
    if len(diff_text) > 6000:
        diff_text = diff_text[:6000] + "\n... (truncated)"

    prompt = VALIDATE_AND_REGENERATE_PROMPT.format(
        smell_type=smell_type,
        smell_description=smell_desc,
        smell_file=smell_file,
        smell_class=smell_class,
        smell_method=smell_method,
        changed_files=", ".join(changed_files),
        smell_diff=diff_text,
        hint_targeted=_get_targeted(entry),
        hint_guided=_get_guided(entry),
        hint_open=_get_open(entry),
    )

    result = call_llm(prompt, model=model, max_tokens=2048, base_url=base_url)
    raw = result.get("raw", "")

    return {
        "targeted_ok": (_parse_xml_tag(raw, "targeted_ok") or "").strip().lower() == "true",
        "targeted_reason": _parse_xml_tag(raw, "targeted_reason") or "",
        "targeted_new": _parse_xml_tag(raw, "targeted_new") or "",
        "guided_ok": (_parse_xml_tag(raw, "guided_ok") or "").strip().lower() == "true",
        "guided_reason": _parse_xml_tag(raw, "guided_reason") or "",
        "guided_new": _parse_xml_tag(raw, "guided_new") or "",
        "open_ok": (_parse_xml_tag(raw, "open_ok") or "").strip().lower() == "true",
        "open_reason": _parse_xml_tag(raw, "open_reason") or "",
        "open_new": _parse_xml_tag(raw, "open_new") or "",
        "usage": result.get("usage", {}),
    }


def _build_settings_from_repo_spec(repo_spec: Dict, repo_name: str = "") -> Dict:
    """Build a settings dict from repo_list.json spec fields."""
    src_path = repo_spec.get("src_path", "")
    envs = dict(repo_spec.get("envs", {}))
    if src_path and repo_name and src_path != repo_name:
        envs.setdefault("PYTHONPATH", str(Path(src_path).parent))
    return {
        "src_path": src_path,
        "commit_id": repo_spec.get("commit_id", ""),
        "test_cmd": repo_spec.get("test_cmd", ""),
        "envs": envs,
        "env_name": repo_spec.get("env_name", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate and regenerate smell instructions in code_smells.json."
    )
    parser.add_argument("--repo-name", required=True, help="Repository name (e.g. click).")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-5-20250929",
                        help="Model for validation and regeneration.")
    parser.add_argument("--base-url", default=None, help="Base URL for API.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate, do not regenerate or update file.")
    parser.add_argument("--force", action="store_true",
                        help="Skip validation, regenerate all entries.")
    args = parser.parse_args()

    # Load code_smells.json
    code_smells_path = os.path.join(args.output_dir, args.repo_name, "code_smells.json")
    if not os.path.exists(code_smells_path):
        print(f"Error: {code_smells_path} not found")
        sys.exit(1)

    with open(code_smells_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from {code_smells_path}")

    # Load repo_list.json for backfilling settings
    repo_spec: Dict = {}
    repo_list_path = "repo_list.json"
    if os.path.exists(repo_list_path):
        with open(repo_list_path, "r", encoding="utf-8") as f:
            repo_list = json.load(f)
        repo_spec = repo_list.get(args.repo_name, {})

    # Backfill settings for entries that are missing it
    settings_backfilled = 0
    for entry in entries:
        if not entry.get("settings") and repo_spec:
            entry["settings"] = _build_settings_from_repo_spec(repo_spec, args.repo_name)
            settings_backfilled += 1
    if settings_backfilled:
        print(f"Backfilled settings for {settings_backfilled} entries from repo_list.json")

    # Load smell type descriptions
    smell_types_map: Dict[str, Dict] = {}
    if os.path.exists("smell_type.json"):
        with open("smell_type.json", "r", encoding="utf-8") as f:
            for s in json.load(f):
                smell_types_map[s["type"]] = s

    modified = settings_backfilled > 0
    total_validated = 0
    total_failed = 0
    total_regenerated = 0

    for idx, entry in enumerate(entries):
        instance_id = entry.get("instance_id", f"entry_{idx}")
        smell_type = entry.get("type", "?")
        difficulty = entry.get("difficulty", "?")
        label = f"[{idx + 1}/{len(entries)}] {instance_id}"

        targeted = _get_targeted(entry)
        guided = _get_guided(entry)
        open_hint = _get_open(entry)

        if not targeted and not guided and not open_hint:
            print(f"  {label}: SKIP (no instruction fields)")
            continue

        # --- Validate and regenerate in one call ---
        print(f"  {label}: validating ...")
        total_validated += 1

        result = validate_and_regenerate_entry(
            entry, smell_types_map, model=args.model, base_url=args.base_url
        )

        # Check if all passed
        if result["targeted_ok"] and result["guided_ok"] and result["open_ok"]:
            print(f"    PASS")
            continue

        # Report failures with reasons
        total_failed += 1
        if not result["targeted_ok"]:
            reason = result.get("targeted_reason", "unknown")
            print(f"    targeted FAIL: {reason}")
        if not result["guided_ok"]:
            reason = result.get("guided_reason", "unknown")
            print(f"    guided FAIL: {reason}")
        if not result["open_ok"]:
            reason = result.get("open_reason", "unknown")
            print(f"    open FAIL: {reason}")

        if args.dry_run:
            continue

        # Apply regenerated hints
        any_changed = False
        if result["targeted_new"]:
            _set_targeted(entry, result["targeted_new"])
            print(f"    new targeted: {result['targeted_new'][:80]}...")
            any_changed = True
        if result["guided_new"]:
            _set_guided(entry, result["guided_new"])
            print(f"    new guided:   {result['guided_new'][:80]}...")
            any_changed = True
        if result["open_new"]:
            _set_open(entry, result["open_new"])
            print(f"    new open:     {result['open_new'][:80]}...")
            any_changed = True

        if any_changed:
            modified = True
            total_regenerated += 1
            # Save immediately after each case
            with open(code_smells_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"    Saved to {code_smells_path}")

    # --- Summary ---
    print(f"\nSummary: {total_validated} validated, {total_failed} failed, "
          f"{total_regenerated} regenerated")

    if args.dry_run:
        print("(dry-run, no changes written)")
    elif modified:
        print(f"All changes saved to {code_smells_path}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
