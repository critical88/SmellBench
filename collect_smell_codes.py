"""
Collect per-repo code_smells.json files and merge them into output/smell_codes.json.

Reads repo_list.json, finds repos marked as selected, loads each repo's
output/{repo_name}/code_smells.json, and writes the combined list to
output/smell_codes.json.

## Selection Logic

**Default behavior** (no --project-name):
- Only processes repos with "selected": true in repo_list.json
- Applies language filter if specified

**With --project-name**:
- Processes specified projects regardless of "selected" status
- User's explicit choice overrides repo_list.json settings
- Still applies language filter if specified

## Usage Examples

    python collect_smell_codes.py
    # Collects all repos with selected=true and language=python (default)

    python collect_smell_codes.py --language python
    # Explicitly filter for Python repos (same as default)

    python collect_smell_codes.py --language all
    # Collects all selected repos regardless of language

    python collect_smell_codes.py --project-name click --project-name flask
    # Collects only click and flask, ignoring selected status
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-repo code_smells.json into a single smell_codes.json."
    )
    parser.add_argument(
        "--repo-list", default="repo_list.json",
        help="Path to repo_list.json (default: repo_list.json).",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory containing per-repo results (default: output).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file path (default: {output-dir}/smell_codes.json).",
    )
    parser.add_argument(
        "--language", default="python",
        help="Only collect repos with this language (default: python). Use 'all' or empty string to collect all languages.",
    )
    parser.add_argument(
        "--project-name", action="append", dest="project_names",
        help="Only collect specific project(s), can be specified multiple times.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    # Normalize language filter: treat 'all' or empty string as no filter
    language_filter = args.language
    if language_filter and language_filter.lower() in ("all", ""):
        language_filter = None

    # Generate output filename with language suffix if specified
    if args.out:
        out_path = args.out
    else:
        if language_filter:
            filename = f"smell_codes_{language_filter.lower()}.json"
        else:
            filename = "smell_codes.json"
        out_path = os.path.join(output_dir, filename)

    # Load repo list
    with open(args.repo_list, "r", encoding="utf-8") as f:
        repo_list = json.load(f)

    all_entries = []
    processed_repos = []
    skipped_repos = []

    for repo_name, spec in repo_list.items():
        # Filter by specific project names if provided
        if args.project_names and repo_name not in args.project_names:
            continue

        # Check if repo is selected (skip this check if user explicitly specified project names)
        if not args.project_names and not spec.get("selected", False):
            skipped_repos.append((repo_name, "not selected"))
            continue

        # Filter by language if specified
        if language_filter:
            repo_lang = spec.get("language", "python").lower()  # Default to python if not specified
            if repo_lang != language_filter.lower():
                skipped_repos.append((repo_name, f"language mismatch ({repo_lang})"))
                continue

        code_smells_path = os.path.join(output_dir, repo_name, "code_smells.json")
        if not os.path.exists(code_smells_path):
            skipped_repos.append((repo_name, "code_smells.json not found"))
            print(f"[skip] {repo_name}: {code_smells_path} not found")
            continue

        with open(code_smells_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # Filter out entries without instance_id (incomplete entries)
        original_count = len(entries)
        entries = [e for e in entries if e.get("instance_id")]
        if len(entries) < original_count:
            skipped_count = original_count - len(entries)
            print(f"[filter] {repo_name}: skipped {skipped_count} incomplete entries (no instance_id)")

        # Add language field to each entry's settings from repo metadata
        repo_language = spec.get("language", "python")  # Default to python if not specified
        for entry in entries:
            # Ensure settings dict exists
            if "settings" not in entry:
                entry["settings"] = {}
            # Add language field if not already present
            if "language" not in entry["settings"]:
                entry["settings"]["language"] = repo_language

        # Additional filtering: filter entries by language
        # If entry has no language field, inherit from repo metadata
        if language_filter:
            original_count = len(entries)
            filtered_entries = []
            for e in entries:
                entry_lang = e.get("language", "")
                # If entry has no language, use repo's language
                if not entry_lang:
                    entry_lang = spec.get("language", "python")

                if entry_lang.lower() == language_filter.lower():
                    filtered_entries.append(e)

            entries = filtered_entries
            if len(entries) < original_count:
                print(f"[filter] {repo_name}: filtered {original_count} -> {len(entries)} entries by language")

        print(f"[load] {repo_name}: {len(entries)} entries")
        processed_repos.append((repo_name, len(entries)))
        all_entries.extend(entries)

    # Write merged output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print("Collection Summary:")
    print("="*60)
    print(f"Processed repos: {len(processed_repos)}")
    for repo, count in processed_repos:
        print(f"  - {repo}: {count} entries")

    if skipped_repos:
        print(f"\nSkipped repos: {len(skipped_repos)}")
        for repo, reason in skipped_repos:
            print(f"  - {repo}: {reason}")

    if language_filter:
        print(f"\nLanguage filter: {language_filter}")
    else:
        print(f"\nLanguage filter: all languages")

    print(f"\nTotal: {len(all_entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
