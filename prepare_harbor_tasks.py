"""Download SmellBench dataset from HuggingFace and convert to harbor dataset.

Usage:
    # Download only (save to smell_codes.json)
    python download_and_convert.py

    # Download and run harbor_adapter to generate tasks
    python download_and_convert.py --task-dir ./output/harbor_tasks

    # Specify output path for smell_codes.json
    python download_and_convert.py --output ./data/smell_codes.json

    # Run harbor_adapter with additional options
    python download_and_convert.py --task-dir ./output/harbor_tasks --hint-type targeted --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def download_smellbench(output_path: str = "smell_codes.json") -> list[dict]:
    """Download SmellBench dataset from HuggingFace and save as smell_codes.json.

    Args:
        output_path: Path to save the smell_codes.json file.

    Returns:
        List of records from the dataset.
    """
    print("Downloading SmellBench dataset from HuggingFace...")
    ds = load_dataset("critical88/SmellBench")

    # Get the split (usually 'train' or default split)
    split_name = list(ds.keys())[0]
    data = ds[split_name]

    # Convert to list of dicts
    records = []
    for item in data:
        record = dict(item)
        records.append(record)

    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Downloaded {len(records)} instances to {output_file}")
    return records


def run_harbor_adapter(
    data_path: str,
    task_dir: str,
    hint_type: str = "",
    limit: int | None = None,
    difficulty: str | None = None,
    overwrite: bool = False,
    model_name: str = "anthropic/claude-sonnet-4-5-20250929",
    skip_judge: bool = False,
) -> None:
    """Run harbor_adapter/run_adapter.py to generate harbor tasks.

    Args:
        data_path: Path to smell_codes.json.
        task_dir: Output directory for harbor tasks.
        hint_type: Hint type for instruction (targeted/guided/empty for both).
        limit: Max number of instances to convert.
        difficulty: Filter by difficulty level.
        overwrite: Overwrite existing task directories.
        model_name: LLM judge model name.
        skip_judge: Skip LLM-as-judge evaluation.
    """
    # Import run_adapter module
    harbor_adapter_dir = Path(__file__).parent / "harbor_adapter"
    sys.path.insert(0, str(harbor_adapter_dir))

    from harbor_adapter.run_adapter import SmellCodeToHarbor, _short_name

    print(f"Converting instances to Harbor tasks in {task_dir}...")

    conv = SmellCodeToHarbor(
        harbor_tasks_root=Path(task_dir),
        data_path=data_path,
        hint_type=hint_type,
        model_name=model_name,
        skip_judge=skip_judge,
    )

    # Get all instance IDs
    ids = conv.get_all_ids()

    # Filter by difficulty if specified
    if difficulty is not None:
        filtered_ids = []
        for iid in ids:
            rec = conv.loader.load(iid)
            if rec.difficulty == difficulty:
                filtered_ids.append(iid)
        ids = filtered_ids
        print(f"Filtered to {len(ids)} instances with difficulty={difficulty}")

    # Limit if specified
    if limit is not None:
        ids = ids[:limit]
        print(f"Limited to {len(ids)} instances")

    # Generate tasks
    if hint_type:
        # Single hint type
        def name_fn(iid: str) -> str:
            rec = conv.loader.load(iid)
            return _short_name(hint_type, rec.difficulty or "medium", rec.type, iid, rec.project_name)

        ok, bad = conv.generate_many(ids, name_fn=name_fn, overwrite=overwrite, hint_type=hint_type)
    else:
        # Both variants
        import random
        ok: list = []
        bad: list = []
        for ht in ("targeted", "guided"):
            def make_name_fn(ht_val: str):
                def name_fn(iid: str) -> str:
                    rec = conv.loader.load(iid)
                    return _short_name(ht_val, rec.difficulty or "medium", rec.type, iid, rec.project_name)
                return name_fn
            o, b = conv.generate_many(ids, name_fn=make_name_fn(ht), overwrite=overwrite, hint_type=ht)
            ok.extend(o)
            bad.extend(b)

    print(f"\nDone. Success: {len(ok)}, Failures: {len(bad)}")
    if bad:
        print("Failures:")
        for iid, reason in bad:
            print(f"  - {iid}: {reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Download SmellBench dataset and optionally run harbor_adapter"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="smell_codes.json",
        help="Output path for smell_codes.json (default: smell_codes.json)",
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help="Output directory for Harbor tasks. If not specified, only downloads the dataset.",
    )
    parser.add_argument(
        "--hint-type",
        type=str,
        default="",
        choices=["", "targeted", "guided"],
        help="Hint type for instruction. Empty generates both variants.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of instances to convert",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard", "expert"],
        help="Filter instances by difficulty",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task directories",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="LLM judge model name",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation",
    )

    args = parser.parse_args()

    # Download dataset
    records = download_smellbench(args.output)

    # Run harbor adapter if task-dir is specified
    if args.task_dir:
        run_harbor_adapter(
            data_path=args.output,
            task_dir=args.task_dir,
            hint_type=args.hint_type,
            limit=args.limit,
            difficulty=args.difficulty,
            overwrite=args.overwrite,
            model_name=args.model_name,
            skip_judge=args.skip_judge,
        )
    else:
        print("\nDataset downloaded successfully.")
        print(f"To generate Harbor tasks, run:")
        print(f"  python download_and_convert.py --task-dir ./output/harbor_tasks")


if __name__ == "__main__":
    main()