import ast
from collections import defaultdict
import os
import re
import sys
import random

from typing import Any, Dict, List, Tuple
from analyzer import MethodAnalyzer
import json
import argparse
from tqdm import tqdm
from utils import get_spec, prepare_to_run
from testunits import replace_and_test_caller, reset_repository

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

def process_refactoring(project_name):
    # Define paths
    base_dir = "../"
    refactor_json_path = os.path.join( 'output', project_name, 'refactor_codes.json')
    base_project_path = os.path.join(base_dir, 'project')
    success_refactor_json_path = os.path.join('output', project_name, 'successful_refactor_codes.json')
    project_path = os.path.join(base_project_path, project_name)

    # Check if paths exist
    if not os.path.exists(refactor_json_path):
        print(f"Refactor JSON not found for {project_name}")
        return False
    if not os.path.exists(project_path):
        print(f"Project directory not found for {project_name}")
        return False

    # Read refactoring JSON
    try:
        with open(refactor_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            settings = json_data.get("settings", {})
            refactor_data = json_data.get("refactor_codes", [])
    except Exception as e:
        print(f"Error reading refactor JSON for {project_name}: {e}")
        return False

    src_path = settings.get("src_path", "")
    test_cmd = settings.get("test_cmd", "")
    envs = settings.get("envs", {})
    # Process each refactoring
    successed_refactor_data = []
    spec = get_spec(project_name)
    if not prepare_to_run(spec):
        print("failed to prepare repo env")
        return False
    for refactor_item in tqdm(refactor_data, desc="testing cases..."):
        success = replace_and_test_caller(
            project_name=project_name, 
            src_path=src_path, 
            testsuites=refactor_item['testsuites'], 
            caller_file_content=refactor_item['caller_file_content'],
            envs=envs,
            test_cmd=test_cmd
        )
        
        if success:
            successed_refactor_data.append(refactor_item)
    print("Number of successful refactorings:", len(successed_refactor_data))
    with open(success_refactor_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "name": project_name,
            "settings": settings,
            "refactor_codes": successed_refactor_data
        }, f, indent=4)
    reset_repository(project_path)
    return successed_refactor_data

def main(args):
    random.seed(args.seed)
    project_name = args.project_name
    
    project_path = args.project_dir
    output_path = args.output_dir
    project_lib = f"{project_path}/{project_name}"

    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
   

    # print(f"Traversing All testunit")
    # generate_function_mapping(project_name, project_path)
    print(f"\nAnalyzing codebase: {project_lib}")
    analyzer = MethodAnalyzer(project_name, project_path, long_method_depth=args.long_method_depth)
    
    result = analyzer.find_refactor_codes()
    
    if not result:
        print("No methods found or analysis failed")
        return
    refactor_codes = result['refactor_codes']
    
    output_base = os.path.join(output_path, project_name)
    os.makedirs(output_base, exist_ok=True)
    with open(os.path.join(output_base, 'refactor_codes.json'), 'w', encoding='utf-8') as f:
        settings = analyzer.meta_info
        saved_json = {
            "name": project_name,
            "settings": settings,
            "refactor_codes": refactor_codes
        }
        json.dump(saved_json, f, indent=2)
    caller_files_dir = os.path.join(output_base, 'caller_files')
    saved_files = save_caller_file_contents(refactor_codes, caller_files_dir)
    print(f"Saved {len(saved_files)} caller file copies to {caller_files_dir}")

    print("Start testunit to filter the illegal code")
    passed_refactors = process_refactoring(project_name)
    print(f"Number of refactor_codes: {result['stat']['raw_refacoter_num']}")
    print(f"Number of refactors with testunits: {result['stat']['refactor_with_test_num']}")
    print("Number of refactor_codes splits: " + str(result['stat']['split']))
    print("Number of passed refactors:", len(passed_refactors))
    stat = defaultdict(dict)
    for refactor in passed_refactors:
        if 'total' not in stat[refactor['type']]:
            stat[refactor['type']]['total'] = 0
        stat[refactor['type']]['total']+=1
        if "key" in refactor['meta']:
            if refactor['meta']['key'] not in stat[refactor['type']]:
                stat[refactor['type']][refactor['meta']['key']] = 0
            stat[refactor['type']][refactor['meta']['key']] += 1
        
    print("Number of passed refactors splits: " + str(dict(stat)))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM refactor ability against reference data.")
    parser.add_argument("--output-dir", default="output", help="Directory for cached outputs and reports.")
    parser.add_argument("--project-dir", default="../project", help="Project directory for resolving relative paths in test commands.")
    parser.add_argument("--project-name", default="click", help="Project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--long-method-depth", type=int, default=3, help="Max callee layers to inline for long-method expansion (None means unlimited).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
