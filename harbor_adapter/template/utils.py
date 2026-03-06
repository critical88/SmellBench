import ast
import textwrap
from codebleu import calc_codebleu
from typing import List, Set, Tuple, Optional, Dict, Any, Sequence
import contextlib
import os
from pathlib import Path
import shlex
import dataclasses
import json
import subprocess


CODE_TEXT_FIELDS = ["code", "source", "body", "text", "snippet"]

@dataclasses.dataclass
class Segment:
    name: str
    text: str
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text

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

def iterify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(iterify(item))
        return flattened
    return [value]

def convert_to_segments(value: Any, fallback_label: str, text_fields: Sequence[str]) -> List[Segment]:
    segments: List[Segment] = []
    for item in iterify(value):
        name = None
        if isinstance(item, str):
            text = item
            meta: Dict[str, Any] = {"source": fallback_label}
        elif isinstance(item, dict):
            text = ""
            for field in text_fields:
                candidate = item.get(field)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break
            if not text:
                text = json.dumps(item, ensure_ascii=False, sort_keys=True)
            meta = item
            if "position" in meta:
                name = meta['position']['method_name']
        else:
            text = str(item)
            meta = {"source": fallback_label}
        text = text.strip()
        if text:
            segments.append(Segment(text=text, meta=meta, name=name))
    return segments

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

def codebleu_similarity(pred: str, ref: str) -> float:
    pred = strip_python_comments(pred)
    ref = strip_python_comments(ref)
    if not pred.strip() and not ref.strip():
        return 0.0
    result = calc_codebleu([pred], [ref], lang='python')
    return result['codebleu']


@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)



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

def _module_relative_path(module_path: str, src_path:str) -> Path:
    cleaned = (module_path or "").strip(".")
    src_tail = Path(src_path).parts[-1] if Path(src_path).parts else src_path
    if cleaned.startswith(f"{src_tail}."):
        cleaned = cleaned[len(src_tail) + 1 :]
    if not cleaned:
        rel = Path("__init__.py")
    else:
        rel = Path(cleaned.replace(".", os.sep) + ".py")
    return Path(src_path) / rel


def _modules_from_paths(src_path, project_path, diff_files: List[str]) -> Set[str]:
    modules: Set[str] = set()
    for diff_file in diff_files:
        module_name = _module_from_diff_path(src_path, project_path, diff_file)
        if module_name:
            modules.add(module_name)
    return modules



def _module_from_diff_path(src_path, project_path, diff_file: str) -> Optional[str]:
    project_repo = project_path
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



def compute_prf(match_count: int, pred_total: int, ref_total: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if pred_total == 0 and ref_total == 0:
        return None, None, None
    precision = match_count / pred_total if pred_total else 0.0
    recall = match_count / ref_total if ref_total else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1