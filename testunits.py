import json
import os
import subprocess
from pathlib import Path
import argparse
from utils import pushd, _log, create_test_command, DEBUG_LOG_LEVEL, _run_git_command, conda_exec_cmd, get_spec
from tqdm import tqdm
import uuid

def reset_repository(repo_path, commit_hash=None):
    """Reset the git repository to a clean state at the given commit."""
    try:
        commit_hash = 'HEAD' if commit_hash is None else commit_hash
        subprocess.run(['git', 'reset', '--hard', commit_hash], cwd=repo_path, check=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error resetting repository at {repo_path}")
        return False


# ---------------------------------------------------------------------------
# Java (Maven) helpers
# ---------------------------------------------------------------------------

def _build_mvn_test_cmd(test_file_paths, test_cmd=""):
    """Build a Maven test command from test file paths.

    test_file_paths can be:
      - test class simple names: ["FileUtilsTest", "IOCaseTest"]
      - test class FQNs: ["org.apache.commons.io.FileUtilsTest"]
      - test class + method: ["FileUtilsTest#testCopy"]
    """
    cmd = ["mvn", "test"]

    if test_file_paths:
        test_pattern = ",".join(test_file_paths)
        cmd.append(f"-Dtest={test_pattern}")

    cmd.append("-DfailIfNoTests=false")
    cmd.append("-Dmaven.test.failure.ignore=false")
    cmd.append("-pl")
    cmd.append(".")

    if test_cmd:
        cmd.extend(test_cmd.split())

    return cmd


def _run_java_tests(project_name, project_path, test_file_paths, envs={}, test_cmd="", timeout=None):
    """Run the Java project's test suite via Maven."""
    try:
        with pushd(project_path):
            exec_env = os.environ.copy()
            for k, v in envs.items():
                exec_env[k] = v

            cmd = _build_mvn_test_cmd(test_file_paths, test_cmd=test_cmd)
            _log(f"Running: {' '.join(cmd)}", DEBUG_LOG_LEVEL)

            result = subprocess.run(
                cmd,
                cwd=".",
                capture_output=True,
                text=True,
                env=exec_env,
                timeout=timeout,
            )
            if result.returncode != 0:
                return False, result.stdout + "\n" + result.stderr

            return True, result.stdout

    except subprocess.TimeoutExpired:
        print(f"Test execution timed out after {timeout}s")
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        print(f"Error running tests: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# Python (pytest / conda) helpers
# ---------------------------------------------------------------------------

def _run_python_tests(project_name, project_path, test_file_paths, envs={}, test_cmd="", timeout=None):
    """Run the Python project's test suite via pytest + conda."""
    spec = get_spec(project_name)
    try:
        with pushd(project_path):
            exec_env = os.environ.copy()
            for k, v in envs.items():
                exec_env[k] = v

            cmd = create_test_command(test_file_paths, test_cmd=test_cmd)
            result = conda_exec_cmd(cmd, spec=spec, cwd=".", envs=exec_env, capture_output=True, timeout=timeout)
            if result.returncode != 0:
                return False, result.stdout

            return True, result.stdout

        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"Error running tests: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------

def _is_java_project(project_name):
    """Check if a project is a Java project based on repo_list.json spec."""
    spec = get_spec(project_name)
    if spec is None:
        return False
    return spec.get("language", "").lower() == "java"


def run_project_tests(project_name, project_path, test_file_paths, envs={}, test_cmd="", timeout=None):
    """Run tests — dispatches to Maven or pytest based on project language."""
    if _is_java_project(project_name):
        return _run_java_tests(project_name, project_path, test_file_paths,
                               envs=envs, test_cmd=test_cmd, timeout=timeout)
    else:
        return _run_python_tests(project_name, project_path, test_file_paths,
                                 envs=envs, test_cmd=test_cmd, timeout=timeout)


def replace_and_test_caller(
    project_name: str,
    src_path: str,
    testsuites,
    smell_content=None,
    test_cmd="",
    envs={},
    project_dir="../project",
    commit_hash=None,
    verbose=False,
):
    """Apply a diff to a project and run the mapped tests.

    Works for both Python and Java projects — test execution is dispatched
    based on the project's language field in repo_list.json.

    1. Reset repository to commit_hash
    2. Apply smell_content as a git diff
    3. Run the specified tests
    4. Reset repository back (in finally)
    """
    base_project_path = project_dir
    project_path = os.path.join(base_project_path, project_name)

    if not os.path.exists(project_path):
        print(f"Project directory not found for {project_name}")
        return False

    if not reset_repository(project_path, commit_hash):
        print(f"Project reset failed")
        return False

    try:
        test_file_paths = testsuites

        if smell_content is not None:
            uuid_str = str(uuid.uuid4())
            try:
                diff_file = os.path.join(project_path, f"{project_name}_smell_{uuid_str}.diff")
                with open(diff_file, "w") as f:
                    f.write(smell_content)
                diff_file_path = os.path.abspath(diff_file)
                _run_git_command(["apply", diff_file_path], cwd=project_path)
            except Exception as e:
                print(f"Error applying smell diff: {e}")
                return False
            finally:
                if os.path.exists(diff_file):
                    os.remove(diff_file)

        # Dispatch timeout: Java projects need more time for Maven
        default_timeout = 120 if _is_java_project(project_name) else 20
        success, output = run_project_tests(
            project_name,
            project_path,
            test_file_paths,
            envs=envs,
            test_cmd=test_cmd,
            timeout=default_timeout,
        )

        if success:
            _log(f"Tests passed for {project_name}")
            return True
        else:
            _log(f"Tests failed for {project_name}")
            _log(f"Output: {output}")
            return False
    finally:
        reset_repository(project_path, commit_hash)
