"""
Collect per-repo code_smells.json files and merge them into output/smell_codes.json.

Reads repo_list.json, finds repos marked as selected, loads each repo's
output/{repo_name}/code_smells.json, and writes the combined list to
output/smell_codes.json.
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
    args = parser.parse_args()

    output_dir = args.output_dir
    out_path = args.out or os.path.join(output_dir, "smell_codes.json")

    # Load repo list
    with open(args.repo_list, "r", encoding="utf-8") as f:
        repo_list = json.load(f)

    all_entries = []

    for repo_name, spec in repo_list.items():
        if not spec.get("selected", False):
            continue

        code_smells_path = os.path.join(output_dir, repo_name, "code_smells.json")
        if not os.path.exists(code_smells_path):
            print(f"[skip] {repo_name}: {code_smells_path} not found")
            continue

        with open(code_smells_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        print(f"[load] {repo_name}: {len(entries)} entries")
        all_entries.extend(entries)

    # Write merged output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\nMerged {len(all_entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
