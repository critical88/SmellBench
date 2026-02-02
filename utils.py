import contextlib
from pathlib import Path
import os
import textwrap
import ast
import shutil
import logging
import uuid
import hashlib
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
INFO_LOG_LEVEL=logging.INFO
DEBUG_LOG_LEVEL=logging.DEBUG

with open("repo_list.json") as f:
    repo_list = json.load(f)

@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

@contextlib.contextmanager
def disableGitTools(project_path: Path):
    if isinstance(project_path, str):
        project_path = Path(project_path)
    git_dir = project_path / ".git"
    tmp_dir = project_path.parent / f"{project_path.name}_{str(uuid.uuid4())}"
    dest_dir = tmp_dir / ".git"
    os.makedirs(tmp_dir, exist_ok=True)
    move = False
    if git_dir.exists():
        shutil.move(git_dir, dest_dir)
        move = True
    try:
        yield
    finally:
        if dest_dir.exists() and move:
            shutil.move(dest_dir, git_dir)
        os.removedirs(tmp_dir)

def _log(text, level=INFO_LOG_LEVEL):
    logging.log(level, text)
    


def hashcode(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def strip_python_comments(text: str) -> str:
    stripped = textwrap.dedent(text)
    if not stripped.strip():
        return stripped
    try:
        tree = ast.parse(stripped)
        method_ast = tree.body[0]
        body_without_docstring = [node for node in method_ast.body if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Str)]
        return ast.unparse(body_without_docstring)
        
    except Exception:
        return text
    
def list_conda_envs():
    process = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    env_list = {}
    for env in process.stdout.split("\n"):
        if env.startswith("#") or not env:
            continue
        env = env.split(" ")
        env_list[env[0]] = env[-1]
    return env_list
    

def is_spec_installed(spec):
    return len(list_uninstalled_specs(spec)) == 0

def list_uninstalled_specs(specs):
    if not isinstance(specs, list):
        specs = [specs]
    env_dict = list_conda_envs()
    uninstalled = []
    for spec in specs:
        if "env_name" in spec and spec['env_name'] not in env_dict:
            uninstalled.append(spec)
    return uninstalled

def prepare_env(spec, project_path="../project"):
    repo_name = spec['name']
    if not spec:
        return False
    if is_spec_installed(spec):
        print(f"{repo_name} environment already exists")
        return True
    repo_path = os.path.join(project_path, repo_name)
    if "conda_env_create" in spec:
        conda_env_create_cmd = spec['conda_env_create']
        if isinstance(conda_env_create_cmd, str):
            conda_env_create_cmd = [conda_env_create_cmd]
        print(f"preparing {repo_name} environment  ...")
        for cmd in conda_env_create_cmd:
            process = subprocess.run(cmd.split(" "), input="y\n" * 10, text=True, cwd=repo_path)
            if process.returncode != 0:
                # print(process.stderr)
                return False
    if is_spec_installed(spec):
        print(f"prepare {repo_name} environment success")
        return True
    else:
        print(f"prepare {repo_name} environment failed")
        return False

def get_repo_name(spec):
    repo_name = spec['name']
    if "src_path" in spec:
        repo_name = spec['src_path'].split("/")[-1]
    return repo_name

def conda_exec_cmd(cmds, spec, cwd=None, envs=None, capture_output=False, use_shell=False, timeout=None):
    env_name = ""
    if "env_name" in spec:
        env_name = spec['env_name']
    prefix = []
    if env_name:
        prefix = ["conda", "run", "-n" , env_name, '--live-stream']
    
    if len(cmds) > 0:
        if isinstance(cmds[0], str):
            cmds = [cmds]
    for cmd in cmds:
        final_cmd = prefix + cmd
        final_cmd = [cmd for cmd in final_cmd if cmd]
        if use_shell:
            final_cmd = " ".join(final_cmd)
        process = subprocess.run(final_cmd, text=True, capture_output=capture_output, cwd=cwd, env=envs, shell=use_shell, timeout=timeout)
        if process.returncode != 0:
            return process
    return process

def _run_git_command(args: Sequence[str], check: bool = True, cwd=None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Git command {' '.join(args)} failed with code {result.returncode}: {result.stderr.strip()}"
        )
    return result

def install_repo(spec, project_path="../project"):
    repo_name = spec['name']

    if isinstalled(spec):
        print(f"{repo_name} already installed")
        return True
    cwd = os.path.join(project_path, repo_name)
    process = _run_git_command(["checkout", spec['commit_id']], cwd=cwd)
    if process.returncode == 0:
        print(f"checkout {repo_name} success")
    
    process = _run_git_command(["clean", "-xdf"], cwd=cwd)
    if process.returncode == 0:
        print(f"clean {repo_name} success")

    build_cmd = ["pip install -e ."]
    if "build_cmd" in spec:
        build_cmd = spec['build_cmd']
    if isinstance(build_cmd, str):
        build_cmd = [build_cmd]

    build_cmd = [bc.split(" ") if isinstance(bc, str) else bc for bc in build_cmd]
    print(f"installing {repo_name} ...")
    process = conda_exec_cmd(build_cmd, spec, cwd=cwd, use_shell=True)
    
    if isinstalled(spec):
        print(f"install {repo_name} success")
        return True
    else:
        print(f"install {repo_name} failed")
        return False

def isinstalled(spec):
    repo_name = get_repo_name(spec)
    cmd = f"pip show {repo_name}"
    process = conda_exec_cmd(cmd.split(" "), spec, capture_output=True)
    
    if process.returncode == 0:
        if any([line.startswith("Editable project location") for line in process.stdout.split("\n")]):
            return True
    
    repo_name = spec['name']
    cmd = f"pip show {repo_name}"
    process = conda_exec_cmd(cmd.split(" "), spec, capture_output=True)
    if process.returncode == 0:
        if any([line.startswith("Editable project location") for line in process.stdout.split("\n")]):
            return True
    return False

def download_repo(spec, project_path="../project"):
    repo_name = spec['name']
    repo_path = os.path.join(project_path, spec['name'])
    if os.path.exists(repo_path):
        print(f"Repo {repo_name} already exists")
        return True
    url = spec['url']
    print(f"Cloning {repo_name} ...")
    if "submodule" in spec:
        subprocess.run(["git", "clone", "--recurse-submodules", url, str(repo_path)], check=True, text=True, capture_output=True)
        print(f"Updating submodule in {repo_name}...")
        subprocess.run(["git", "submodule", "update", "--init"], check=True, text=True, cwd=repo_path)
    else:
        subprocess.run(["git", "clone", url, str(repo_path)], check=True, text=True, capture_output=True)
    if os.path.exists(repo_path):
        print(f"Repo {repo_name} clone success")
        return True
    else:
        print(f"Repo {repo_name} clone failed")
        return False

def prepare_to_run(spec, project_path="../project"):
    if not download_repo(spec):
        return False
    if not prepare_env(spec):
        return False
    if not install_repo(spec, project_path=project_path):
        return False
    repo_name = spec['name']
    print(f"repo {repo_name} is already done")
    return True

def get_spec(project_name):
    if project_name not in repo_list:
        return
    repo = repo_list[project_name]
    repo['name'] = project_name
    return repo

# spec = get_spec("pandas")
# ret = prepare_to_run(spec)
# print(ret)