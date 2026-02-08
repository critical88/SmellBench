import json
import os
import subprocess
from pathlib import Path
import argparse
from utils import pushd, _log, DEBUG_LOG_LEVEL, prepare_to_run, conda_exec_cmd, get_spec
from tqdm import tqdm
import shlex

def reset_repository(repo_path, commit_hash=None):
    """Reset the git repository to its latest state"""
    try:
        commit_hash = 'HEAD' if commit_hash is None else commit_hash
        subprocess.run(['git', 'reset', '--hard', commit_hash], cwd=repo_path, check=True)
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

def create_test_command( test_file_paths=[], test_cmd="", envs=None):
    cmd = []
    if test_cmd:
        cmd.extend(shlex.split(test_cmd, posix=True))
    
    cmd.extend(test_file_paths)
    if envs is not None:
        cmd = [f"{k}={v}" for k, v in envs.items()] + ["pytest", "-x"] + cmd
    else:
        cmd = ["pytest", "-x"] + cmd
    return cmd

def run_project_tests(project_name, project_path, test_file_paths, envs={}, test_cmd="", timeout=None):
    spec = get_spec(project_name)
    """Run the project's test suite"""
    try:
        # First try to install the project
        # subprocess.run(['pip', 'install', '-e', '.'], cwd=project_path, check=True)
        # Then run tests
        with pushd(project_path):
            exec_env = os.environ.copy()
            for k, v in envs.items():
                exec_env[k] = v
            # batch_size = 300
            # i = 0
            # test_len = len(test_file_paths)
            # if test_len > 300:
            #     ## run all test 
            #     return None, None
            # while(i >= 0 and batch_size * i < test_len):
            # cmd = []
            # if test_cmd:
            #     cmd.extend(shlex.split(test_cmd, posix=True))
            
            # cmd.extend(test_file_paths[batch_size * i: batch_size * (i+1)])
            # i += 1
            cmd = create_test_command(test_file_paths, test_cmd=test_cmd)
            result = conda_exec_cmd(cmd, spec=spec, cwd=".", envs=exec_env, capture_output=True, timeout=timeout)
            # result = subprocess.run(cmd, cwd='.', capture_output=True, text=True, env=env)
            if result.returncode != 0:
                return False, result.stdout
                
            return True, result.stdout

            # test_func = [] if len(test_file_paths) > 100 else test_file_paths
            # if ignore_test:
            #     test_func.extend([f'--ignore={p}' for p in ignore_test])
            # result = subprocess.run(['pytest','-x'] + test_func, cwd='.', capture_output=True, text=True, env=env)
        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"Error running tests: {e}")
        return False, str(e)

def replace_and_test_caller(project_name:str, src_path:str, testsuites, caller_file_content=None, test_cmd="", envs={}, project_dir="../project", commit_hash=None, verbose=False):
    # Define paths
    base_project_path = project_dir
    module_path = os.path.normpath(src_path).split(os.sep)[-1]
    project_path = os.path.join(base_project_path, project_name)
    # Check if paths exist
    if not os.path.exists(project_path):
        print(f"Project directory not found for {project_name}")
        return False

    # Reset repository
    if not reset_repository(project_path, commit_hash):
        print(f"Project reset failed")
        return False
    try:
        test_file_paths = testsuites
        if caller_file_content is not None:
            for code_item in caller_file_content:
                module = code_item.get('module_path', '').lstrip(module_path).lstrip(".")
                file_path = os.path.join(base_project_path, project_name, src_path, module.replace(".", os.sep) + ".py")
                success = replace_file_content(file_path, code_item.get('code', ''))
                if not success:
                    print(f"Failed to replace file {file_path}")
                    return False
        # Run tests
        success, output = run_project_tests(project_name, project_path, test_file_paths, envs=envs, test_cmd=test_cmd, timeout=20)
    
        if success:
            # print(f"Tests passed for {project_name}")
            _log(f"Tests passed for {project_name}")
            return True
        else:
            _log(f"Tests failed for {project_name}")
            _log(f"Output: {output}")
            return False
    finally:
        reset_repository(project_path, commit_hash)

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
