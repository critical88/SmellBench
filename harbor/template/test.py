


from typing import Sequence, Dict, List, Any,Set, Optional
import subprocess
from collections import defaultdict
import os
import json
import dataclasses
from pathlib import Path
import re
import ast
import textwrap
import contextlib
import shlex

def normalize_snippet(snippet: Optional[str]) -> str:
    if snippet is None:
        return ""
    return textwrap.dedent(snippet).strip()

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

def test_and_eval(args):

    smell_commit_file = args.smell
    instance_file = args.instance
    with open(instance_file) as f:
        instance = json.loads(f)
    instance_id = instance['instance_id']
    project_name = instance['name']
    project_repo = f"/workspace/project/{project_name}"

    
    smell_commit_id = None
    with open(smell_commit_file) as f:
        smell_commit_id = f.readline().strip()

    diff_text = _run_git_command(["diff", smell_commit_id], cwd=project_repo).stdout
    diff_output = _run_git_command(["diff", "--name-only", smell_commit_id], cwd=project_repo).stdout
    diff_files = [line.strip() for line in diff_output.splitlines() if line.strip()]


    success, output = run_project_tests(project_name, project_repo, instance["testsuites"], envs=instance['settings']['envs'], test_cmd=instance['settings']['test_cmd'], timeout=60)

    prediction = _build_prediction_from_repo(instance, diff_files, diff_text, response)


@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_project_tests(case, project_name, project_path, test_file_paths, envs={}, test_cmd="", timeout=None):
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


def create_test_command(test_file_paths=[], test_cmd="", envs=None):
    cmd = []
    if test_cmd:
        cmd.extend(shlex.split(test_cmd, posix=True))
    
    cmd.extend(test_file_paths)
    if envs is not None:
        cmd = [f"{k}={v}" for k, v in envs.items()] + ["pytest", "-x"] + cmd
    else:
        cmd = ["pytest", "-x"] + cmd
    return cmd

@dataclasses.dataclass
class Segment:
    name: str
    text: str
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text

@dataclasses.dataclass
class PredictionArtifacts:
    caller_segments: List[Segment]
    callee_segments: List[Segment]
    test_passed: Optional[bool]
    raw_response: str
    parsed_payload: Optional[Dict[str, Any]]
    response: Optional[Dict] = None
    error: Optional[str] = None


@dataclasses.dataclass
class CallSite:
    code: str
    name: str

@dataclasses.dataclass
class FunctionSnippet:
    name: str
    source: str
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)
    calls: List[CallSite] = dataclasses.field(default_factory=list)



def _modules_from_paths( case, diff_files: List[str]) -> Set[str]:
    modules: Set[str] = set()
    for diff_file in diff_files:
        module_name = _module_from_diff_path(case, diff_file)
        if module_name:
            modules.add(module_name)
    return modules

def _build_prediction_from_repo(case, diff_files, diff_text, response):
    project_name = case['name']
    project_repo = f"/workspace/project/{project_name}"
    changed_modules = _modules_from_paths(case, diff_files)
    changed_line_map = _parse_diff_changed_lines(case, diff_text) if diff_text else {}
    target_lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    src_path = case['settings']['src_path']
    for entry in case.get("before_refactor_code", []):
        module_path = entry.get("module_path")
        if module_path:
            target_lookup[module_path].append(entry)
    functions: List[FunctionSnippet] = []
    caller_segments: List[Segment] = []
    callee_segments: List[Segment] = []
    caller_modules: Set[str] = set()
    for file_entry in case.get("caller_file_content", []):
        module_path = file_entry.get("module_path")
        if not module_path:
            continue
        caller_modules.add(module_path)
    candidate_modules: Set[str] = set()
    if changed_line_map:
        candidate_modules.update(module_path for module_path in changed_line_map.keys() if module_path)
    else:
        candidate_modules.update(caller_modules)
        candidate_modules.update(changed_modules)
        for diff_file in diff_files:
            module_path = _module_from_diff_path(case, diff_file)
            if module_path:
                candidate_modules.add(module_path)
    module_file_map: Dict[str, Path] = {}
    for module_path in candidate_modules:
        rel_path = _module_relative_path(module_path, src_path)
        abs_path = project_repo / rel_path
        if not abs_path.exists() or abs_path.suffix.lower() != ".py":
            continue
        module_file_map[module_path] = abs_path
    for module_path, abs_path in module_file_map.items():
        current_code = abs_path.read_text(encoding="utf-8")
        lineno = changed_line_map.get(module_path)
        active_functions = _extract_functions_from_block(current_code, lineno=lineno)
        if not active_functions:
            continue
        functions.extend(active_functions)
        target_entries = target_lookup.get(module_path, [])
        target_names = {entry.get("method_name") for entry in target_entries}
        for entry in target_entries:
            match = _match_function(entry, active_functions)
            if match:
                meta = {
                    "type": "caller",
                    "position": {
                        "module_path": module_path,
                        "class_name": entry.get("class_name"),
                        "method_name": entry.get("method_name"),
                    },
                    "callees": [],
                }
                caller_segments.append(Segment(text=match.source, meta=meta, name=match.name))
        for fn in active_functions:
            simple_name = fn.name.split(".")[-1]
            if simple_name in target_names:
                continue
            position = {
                "module_path": module_path,
                "class_name": fn.meta.get("class_name"),
                "method_name": simple_name,
            }
            callee_segments.append(
                Segment(
                    text=fn.source,
                    meta={
                        "type": "callee",
                        "module_path": module_path,
                        "position": position,
                    },
                    name=simple_name,
                )
            )
    return PredictionArtifacts(
        # functions=functions,
        caller_segments=caller_segments,
        callee_segments=callee_segments,
        test_passed=None,
        raw_response=diff_text,
        parsed_payload={
            "backend": "code_agent",
            "changed_modules": sorted(changed_modules),
        },
    )


def _module_from_diff_path(self, case, diff_file: str) -> Optional[str]:
    src_path = case['settings']['src_path']
    project_repo = self.get_project_path(case)
    src_root = Path(src_path)
    if not src_root.is_absolute():
        src_root = (project_repo / src_root).resolve()
    else:
        src_root = src_root.resolve()
    rel_path = Path(diff_file.strip())
    if not rel_path.is_absolute():
        rel_path = (project_repo / rel_path).resolve()
    else:
        rel_path = rel_path.resolve()
    try:
        relative = rel_path.relative_to(src_root)
    except ValueError:
        return None
    base_module = src_root.parts[-1] if src_root.parts else src_root.name
    return ".".join([base_module] + list(relative.with_suffix("").parts))

def _parse_diff_changed_lines_manual(self, case, diff_text: str) -> Dict[str, Set[int]]:
    changed_lines: Dict[str, Set[int]] = defaultdict(set)
    current_module: Optional[str] = None
    new_line: Optional[int] = None
    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("diff --git"):
            current_module = None
            new_line = None
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                current_module = None
                continue
            if path.startswith("b/"):
                path = path[2:]
            module_path = self._module_from_diff_path(case, path)
            current_module = module_path
            new_line = None
            continue
        if not current_module:
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                new_line = int(match.group(1))
            else:
                new_line = None
            continue
        if new_line is None:
            continue
        prefix = line[:1]
        if prefix == "+":
            changed_lines[current_module].add(new_line)
            new_line += 1
        elif prefix == "-":
            marker = new_line if new_line is not None else 1
            changed_lines[current_module].add(marker)
            continue
        elif prefix == "\\":
            continue
        else:
            new_line += 1
    return changed_lines

def _parse_diff_changed_lines(self, case, diff_text: str) -> Dict[str, Set[int]]:
    if not diff_text or not diff_text.strip():
        return {}
    return _parse_diff_changed_lines_manual(case, diff_text)


def _match_function(self, target: Dict[str, Any], functions: List[FunctionSnippet]) -> Optional[FunctionSnippet]:
    target_name = target.get("method_name")
    if not target_name:
        return None
    target_class = target.get("class_name")
    for fn in functions:
        components = fn.name.split(".")
        simple_name = components[-1]
        if simple_name != target_name:
            continue
        if target_class:
            if len(components) < 2 or components[-2] != target_class:
                continue
        return fn
    return None

def _extract_functions_from_block(code_block: str, common=False, lineno=None) -> List[FunctionSnippet]:
    if not code_block:
        return []
    code_block = normalize_snippet(code_block)
    try:
        tree = ast.parse(code_block)
    except SyntaxError:
        print("code block could not be parsed:")
        return []
    collector = _FunctionCollector(code_block, lineno=lineno)
    collector.visit(tree)
    functions = collector.functions
    for fn in functions:
        fn.meta['common'] = common
    return functions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Harbor test runner for SmellBench instances")
    parser.add_argument("--smell", type=str, required=True, help="Path to smell file (e.g. smell.json)")
    parser.add_argument("--instance", type=str, required=True, help="Path to instance file (e.g. instance.json)")
    args = parser.parse_args()
    print(f"Smell file: {args.smell}, Instance file: {args.instance}")
    test_and_eval(args)