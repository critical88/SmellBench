import ast
from collections import defaultdict
import os
import re
import sys
import traceback
from typing import Dict, List, Tuple
from analyzer import MethodAnalyzer
import json

def main():
    project_name = "urllib3"
    src_path = "src/urllib3"
    project_path = "../project"
    project_lib = f"{project_path}/{project_name}/{src_path}"

    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"\nAnalyzing codebase: {project_lib}")
    analyzer = MethodAnalyzer(project_name, src_path, project_path)

    
    refactor_codes = analyzer.find_refactor_codes()
    
    if not refactor_codes:
        print("No methods found or analysis failed")
        return
    output_base = os.path.join('output', project_name)
    os.makedirs(output_base, exist_ok=True)
    with open(os.path.join(output_base, 'refactor_codes.json'), 'w', encoding='utf-8') as f:
        json.dump(refactor_codes, f, indent=2)
        

if __name__ == "__main__":
    main()
