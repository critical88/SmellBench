"""
Collect per-repo code_smells.json files and merge them into output/smell_codes.json.

Reads repo_list.json, finds repos marked as selected, loads each repo's
output/{repo_name}/code_smells.json, and writes the combined list to
output/smell_codes.json.

Supports filtering by language to collect only specific language projects.

Usage:
    python collect_smell_codes.py                                      # All selected repos
    python collect_smell_codes.py --language python                    # Only Python repos
    python collect_smell_codes.py --project-name click --project-name flask  # Specific repos only
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
        help="Only collect repos with this language (default: python). Use empty string to collect all languages.",
    )
    parser.add_argument(
        "--project-name", action="append", dest="project_names",
        help="Only collect specific project(s), can be specified multiple times.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    out_path = args.out or os.path.join(output_dir, "smell_codes.json")

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

        # Check if repo is selected
        if not spec.get("selected", False):
            skipped_repos.append((repo_name, "not selected"))
            continue

        # Filter by language if specified
        if args.language:
            repo_lang = spec.get("language", "python").lower()  # Default to python if not specified
            if repo_lang != args.language.lower():
                skipped_repos.append((repo_name, f"language mismatch ({repo_lang})"))
                continue

        code_smells_path = os.path.join(output_dir, repo_name, "code_smells.json")
        if not os.path.exists(code_smells_path):
            skipped_repos.append((repo_name, "code_smells.json not found"))
            print(f"[skip] {repo_name}: {code_smells_path} not found")
            continue

        with open(code_smells_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # Additional filtering: filter entries by language if they have language field
        if args.language:
            original_count = len(entries)
            entries = [
                e for e in entries
                if e.get("language", "").lower() == args.language.lower()
                or not e.get("language")  # Keep entries without language field
            ]
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

    if args.language:
        print(f"\nLanguage filter: {args.language}")

    print(f"\nTotal: {len(all_entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
