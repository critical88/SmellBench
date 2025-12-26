import json
import os
import subprocess
from pathlib import Path
import contextlib
import argparse

def reset_repository(repo_path):
    """Reset the git repository to its latest state"""
    try:
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, check=True)
        # subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error resetting repository at {repo_path}")
        return False

def replace_file_content(file_path, new_content):
    """Replace the content of a file with new content"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Error replacing file {file_path}: {e}")
        return False
    
@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
def run_project_tests(project_path, src_path, test_file_paths):
    """Run the project's test suite"""
    
    try:
        # First try to install the project
        # subprocess.run(['pip', 'install', '-e', '.'], cwd=project_path, check=True)
        # Then run tests
        with pushd(project_path):
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(src_path).parent)

            test_func = [] if len(test_file_paths) > 100 else test_file_paths
            result = subprocess.run(['pytest','-x'] + test_func, cwd='.', capture_output=True, text=True, env=env)
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return False, str(e)

def replace_and_test_caller(project_name:str, src_path:str, testsuites, caller_file_content, poject_dir="../project"):
    # Define paths
    base_project_path = poject_dir
    module_path = os.path.normpath(src_path).split(os.sep)[-1]
    project_path = os.path.join(base_project_path, project_name)
    # Check if paths exist
    if not os.path.exists(project_path):
        print(f"Project directory not found for {project_name}")
        return False

    # Reset repository
    if not reset_repository(project_path):
        print(f"Project reset failed")
        return False

    # Prepare test file paths
    test_file_paths = testsuites
    # for test_module_path in testsuites:
    #     path = test_module_path[0].lstrip(project_name).lstrip(".").replace(".", os.sep) + ".py"
    #     file_path = os.path.join(base_project_path, project_name, path)
    #     if os.path.exists(file_path):
    #         test_file_paths.append(os.path.abspath(file_path))
    #     else:
    #         print(f"Test file not found: {path}")
    #         return False

    # Replace caller files
    for code_item in caller_file_content:
        module_path = code_item.get('module_path', '').lstrip(module_path).lstrip(".")
        file_path = os.path.join(base_project_path, project_name, src_path, module_path.replace(".", os.sep) + ".py")
        success = replace_file_content(file_path, code_item.get('code', ''))
        if not success:
            print(f"Failed to replace file {file_path}")
            return False

    # Run tests
    success, output = run_project_tests(project_path, src_path, test_file_paths)
    if success:
        print(f"Tests passed for {project_name}")
        return True
    else:
        print(f"Tests failed for {project_name}\nOutput: {output}")
        return False

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
    # Process each refactoring
    successed_refactor_data = []
    for refactor_item in refactor_data:
        success = replace_and_test_caller(project_name, src_path, refactor_item['testsuites'], refactor_item['caller_file_content'])
        
        if success:
            successed_refactor_data.append(refactor_item)
    print("Number of successful refactorings:", len(successed_refactor_data))
    with open(success_refactor_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "settings": settings,
            "refactor_codes": successed_refactor_data
        }, f, indent=4)
    reset_repository(project_path)
    return True


def main(args):
    project_name = args.project_name
    # List of projects to process
    projects = [project_name]  # Add more projects as needed
    
    results = {}
    for project in projects:
        print(f"\nProcessing {project}...")
        results[project] = process_refactoring(project)
    
    # Print summary
    print("\nSummary:")
    for project, success in results.items():
        status = "Finished" if success else "Failed"
        print(f"{project}: {status}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM refactor ability against reference data.")
    parser.add_argument("--project-name", default="click", help="Project name")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)