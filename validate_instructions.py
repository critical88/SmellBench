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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALIDATE_PROMPT = """You are a QA reviewer for a code-smell benchmark. Each benchmark entry has two instruction fields that will be given to a coding agent as task descriptions.

## Rules

**hint_targeted** (also called "easy"):
- MUST mention the smell type (e.g. "feature envy", "god class").
- MUST mention at least one specific file path where the smell lives.
- MUST mention the class name and/or method name where the smell is centered.
- Should read as a natural-language task asking the agent to find and refactor the smell.

**hint_guided** (also called "hard"):
- MUST mention the smell type.
- MUST mention at least one related file path.
- MUST NOT reveal specific class names, method names, or line numbers — the agent must discover those itself.
- Should read as a natural-language task asking the agent to find and refactor the smell.

## Entry to validate

- **Smell type**: {smell_type}
- **Smell function (ground truth)**: {smell_function}
- **Files changed in diff**: {changed_files}

### hint_targeted value:
{hint_targeted}

### hint_guided value:
{hint_guided}

## Your Task

Check each field against its rules. Return your result using XML tags:

<targeted_ok>true or false</targeted_ok>
<targeted_issues>If false, explain what is wrong. If true, write "none".</targeted_issues>
<guided_ok>true or false</guided_ok>
<guided_issues>If false, explain what is wrong. If true, write "none".</guided_issues>
"""


REGENERATE_PROMPT = """You are a benchmark author for a code-smell detection benchmark.

Given the following smell injection information, write TWO task instructions for a coding agent.

## Smell Info
- **Smell type**: {smell_type}
- **Smell description**: {smell_description}
- **Smell function (ground truth)**: file=`{smell_file}`, class=`{smell_class}`, method=`{smell_method}`
- **Files changed in diff**: {changed_files}

## Diff
```diff
{smell_diff}
```

## Instructions to write

### hint_targeted
A natural-language task description that tells the agent to identify and refactor the smell.
MUST include: the smell type, the specific file path, class name (if applicable), and method name (if applicable).
Write freely — do NOT follow a fixed template. Be specific enough that the agent knows exactly where to look.

### hint_guided
A natural-language task description that tells the agent to identify and refactor the smell.
MUST include ONLY: the smell type and the related file paths.
Do NOT reveal specific class names, method names, or line numbers — the agent must discover those itself.
Write freely — do NOT follow a fixed template.

## Output Format

<hint_targeted>
Your targeted instruction here.
</hint_targeted>

<hint_guided>
Your guided instruction here.
</hint_guided>
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


def validate_entry(
    entry: Dict,
    model: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate one entry's instructions via LLM.

    Returns dict with keys: targeted_ok, targeted_issues, guided_ok, guided_issues.
    """
    smell_type = entry.get("type", "")
    smell_function = entry.get("smell_function", [])
    changed_files = _extract_changed_files(entry.get("smell_content", ""))

    prompt = VALIDATE_PROMPT.format(
        smell_type=smell_type,
        smell_function=smell_function,
        changed_files=", ".join(changed_files),
        hint_targeted=_get_targeted(entry),
        hint_guided=_get_guided(entry),
    )

    result = call_llm(prompt, model=model, base_url=base_url)
    raw = result.get("raw", "")

    targeted_ok = _parse_xml_tag(raw, "targeted_ok") or ""
    guided_ok = _parse_xml_tag(raw, "guided_ok") or ""

    return {
        "targeted_ok": targeted_ok.strip().lower() == "true",
        "targeted_issues": _parse_xml_tag(raw, "targeted_issues") or "",
        "guided_ok": guided_ok.strip().lower() == "true",
        "guided_issues": _parse_xml_tag(raw, "guided_issues") or "",
        "usage": result.get("usage", {}),
    }


def regenerate_instructions(
    entry: Dict,
    smell_types: Dict[str, Dict],
    model: str,
    base_url: Optional[str] = None,
) -> Tuple[str, str, Dict]:
    """Regenerate both instructions for an entry.

    Returns (hint_targeted, hint_guided, usage).
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

    prompt = REGENERATE_PROMPT.format(
        smell_type=smell_type,
        smell_description=smell_desc,
        smell_file=smell_file,
        smell_class=smell_class,
        smell_method=smell_method,
        changed_files=", ".join(changed_files),
        smell_diff=diff_text,
    )

    result = call_llm(prompt, model=model, max_tokens=2048, base_url=base_url)
    raw = result.get("raw", "")

    hint_targeted = _parse_xml_tag(raw, "hint_targeted") or ""
    hint_guided = _parse_xml_tag(raw, "hint_guided") or ""

    return hint_targeted, hint_guided, result.get("usage", {})


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

        if not targeted and not guided:
            print(f"  {label}: SKIP (no instruction fields)")
            continue

        # --- Force mode: skip validation, regenerate all ---
        if args.force:
            print(f"  {label}: FORCE regenerate ...")
            new_targeted, new_guided, usage = regenerate_instructions(
                entry, smell_types_map, model=args.model, base_url=args.base_url,
            )
            if new_targeted:
                _set_targeted(entry, new_targeted)
            if new_guided:
                _set_guided(entry, new_guided)
            modified = True
            total_regenerated += 1
            print(f"    targeted: {new_targeted[:80]}...")
            print(f"    guided:   {new_guided[:80]}...")
            continue

        # --- Validate ---
        print(f"  {label}: validating ...")
        total_validated += 1
        vresult = validate_entry(entry, model=args.model, base_url=args.base_url)

        if vresult["targeted_ok"] and vresult["guided_ok"]:
            print(f"    PASS")
            continue

        total_failed += 1
        if not vresult["targeted_ok"]:
            print(f"    targeted FAIL: {vresult['targeted_issues']}")
        if not vresult["guided_ok"]:
            print(f"    guided FAIL: {vresult['guided_issues']}")

        if args.dry_run:
            continue

        # --- Regenerate ---
        print(f"    Regenerating ...")
        new_targeted, new_guided, usage = regenerate_instructions(
            entry, smell_types_map, model=args.model, base_url=args.base_url,
        )
        if new_targeted:
            _set_targeted(entry, new_targeted)
            print(f"    new targeted: {new_targeted[:80]}...")
        if new_guided:
            _set_guided(entry, new_guided)
            print(f"    new guided:   {new_guided[:80]}...")
        modified = True
        total_regenerated += 1

    # --- Save ---
    print(f"\nSummary: {total_validated} validated, {total_failed} failed, "
          f"{total_regenerated} regenerated")

    if modified and not args.dry_run:
        with open(code_smells_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"Updated {code_smells_path}")
    elif args.dry_run:
        print("(dry-run, no changes written)")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
