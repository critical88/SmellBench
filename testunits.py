import json
import os
import subprocess
from pathlib import Path

def reset_repository(repo_path):
    """Reset the git repository to its latest state"""
    try:
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, check=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, check=True)
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

def run_project_tests(project_path):
    """Run the project's test suite"""
    try:
        # First try to install the project
        subprocess.run(['pip', 'install', '-e', '.'], cwd=project_path, check=True)
        # Then run tests
        result = subprocess.run(['python', '-m', 'pytest'], cwd=project_path, capture_output=True, text=True)
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return False, str(e)

def process_refactoring(project_name):
    # Define paths
    base_dir = "../"
    refactor_json_path = os.path.join(base_dir, 'analyze_methods', 'output', project_name, 'refactor_codes.json')
    base_project_path = os.path.join(base_dir, 'project')
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
            refactor_data = json.load(f)
    except Exception as e:
        print(f"Error reading refactor JSON for {project_name}: {e}")
        return False


    # Process each refactoring
    for refactor_item in refactor_data:
        # Reset repository
        if not reset_repository(project_path):
            return False
        
        caller_file_content = refactor_item.get('caller_file_content', [])
        
        # Replace caller files
        for code_item in caller_file_content:
            file_path = os.path.join(base_project_path, (code_item.get('module_path', '').replace(".", os.sep) + ".py"))
            success = replace_file_content(file_path, code_item.get('code', ''))
            if not success:
                print(f"Failed to replace file {file_path}")
                return False

        # Run tests
        success, output = run_project_tests(project_path)
        if success:
            print(f"Tests passed for {project_name}")
        else:
            print(f"Tests failed for {project_name}\nOutput: {output}")


def main():
    # List of projects to process
    projects = ['urllib3']  # Add more projects as needed
    
    results = {}
    for project in projects:
        print(f"\nProcessing {project}...")
        results[project] = process_refactoring(project)
    
    # Print summary
    print("\nSummary:")
    for project, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{project}: {status}")

if __name__ == '__main__':
    main()