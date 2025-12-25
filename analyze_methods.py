import ast
from collections import defaultdict
import os
import re
import sys

from typing import Any, Dict, List, Tuple
from analyzer import MethodAnalyzer
import json
import argparse

def save_caller_file_contents(result: Any, output_dir: str) -> List[str]:
    """
    Persist caller_file_content blocks from the analyzer result to disk.

    The function accepts either the in-memory result list, a dict that contains
    the "refactor_codes" key, or a JSON string representation of that data.
    Each caller_file_content entry is written to a file whose path mirrors the
    module_path value (e.g. urllib3.connectionpool -> urllib3/connectionpool.py).
    Later occurrences of the same module_path overwrite earlier ones.
    """
    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, dict):
        refactor_items = result.get("refactor_codes", [])
    else:
        refactor_items = result

    if not isinstance(refactor_items, list):
        raise ValueError("result must be a list or a dict containing 'refactor_codes'")

    saved_files: List[str] = []
    os.makedirs(output_dir, exist_ok=True)

    for item in refactor_items:
        caller_files = item.get("caller_file_content") or []
        for caller_entry in caller_files:
            module_path = (caller_entry.get("module_path") or "").lstrip(".")
            code = caller_entry.get("code")
            if not module_path or code is None:
                continue
            relative_path = module_path.replace(".", os.sep) + "_" + caller_entry['file_suffix']+ ".py"
            file_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
                # if not code.endswith("\n"):
                #     f.write("\n")
            saved_files.append(file_path)

    return saved_files

def main(args):
    projects = {
        "urllib3": {
            "src_path": "src/urllib3"
        },
        "numpy": {
            "src_path": "numpy"
        },
        "requests": {
            "src_path": "src/requests"
        },
        "pydantic": {
            "src_path": "pydantic"
        },
        "click": {
            "src_path": "src/click"
        },
        "jinja": {
            "src_path": "src/jinja2"
        },
    }
    project_name = args.project_name
    src_path = projects[project_name]['src_path']
    project_path = args.project_dir
    output_path = args.output_dir
    project_lib = f"{project_path}/{project_name}/{src_path}"

    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"\nAnalyzing codebase: {project_lib}")
    analyzer = MethodAnalyzer(project_name, src_path, project_path)

    
    refactor_codes = analyzer.find_refactor_codes()
    
    if not refactor_codes:
        print("No methods found or analysis failed")
        return
    output_base = os.path.join(output_path, project_name)
    os.makedirs(output_base, exist_ok=True)
    with open(os.path.join(output_base, 'refactor_codes.json'), 'w', encoding='utf-8') as f:
        settings = {
            "src_path": src_path
        }
        saved_json = {
            "settings": settings,
            "refactor_codes": refactor_codes
        }
        json.dump(saved_json, f, indent=2)
    caller_files_dir = os.path.join(output_base, 'caller_files')
    saved_files = save_caller_file_contents(refactor_codes, caller_files_dir)
    print(f"Saved {len(saved_files)} caller file copies to {caller_files_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM refactor ability against reference data.")
    parser.add_argument("--output-dir", default="output", help="Directory for cached outputs and reports.")
    parser.add_argument("--project-dir", default="../project", help="Project directory for resolving relative paths in test commands.")
    parser.add_argument("--project-name", default="click", help="Project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)