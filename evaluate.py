import argparse
import ast
import dataclasses
import hashlib
import json
import keyword
import os
from pathlib import Path
import uuid
import re
import shlex
import subprocess
import textwrap
from codebleu import calc_codebleu
import time 
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from client import LLMFactory, LLMClient, AgentClient, Client, LLMResponse, AgentResponse
from collections import defaultdict
from testunits import replace_and_test_caller, run_project_tests, create_test_command
from utils import strip_python_comments, disableGitTools,_run_git_command, prepare_to_run, get_spec
try:
    from unidiff import PatchSet
except ImportError:
    PatchSet = None  # type: ignore[assignment]

from prompts import *

CODE_TEXT_FIELDS = ["code", "source", "body", "text", "snippet"]


PYTHON_FENCE_PATTERN = re.compile(r"```python[ \t]*\n([\s\S]*?)```", re.IGNORECASE)
COMMON_FENCE_PATTERN = re.compile(r"```common[ \t]*\n([\s\S]*?)```", re.IGNORECASE)
GENERIC_FENCE_PATTERN = re.compile(r"```([\s\S]*?)```", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\w+|\S", re.UNICODE)
PYTHON_KEYWORDS = set(keyword.kwlist)


def normalize_snippet(snippet: Optional[str]) -> str:
    if snippet is None:
        return ""
    return textwrap.dedent(snippet).strip()


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
class CaseResult:
    instance_id: str
    prompt_hash: str
    caller_accuracy: Optional[float]
    callee_precision: Optional[float]
    callee_recall: Optional[float]
    callee_f1: Optional[float]
    response_stat: Optional[Dict]
    callee_match_score: Optional[float]
    test_passed: Optional[bool]
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        for field in dataclasses.fields(self):
            setattr(self, field.name, kwargs.get(field.name))

def iterify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(iterify(item))
        return flattened
    return [value]


def collect_values_by_key(source: Any, keys: Sequence[str]) -> List[Any]:
    matches: List[Any] = []
    if isinstance(source, dict):
        for key, val in source.items():
            if key in keys:
                matches.append(val)
            matches.extend(collect_values_by_key(val, keys))
    elif isinstance(source, list):
        for item in source:
            matches.extend(collect_values_by_key(item, keys))
    return matches


def collect_by_type(source: Any, desired_type: str) -> List[Any]:
    matches: List[Any] = []
    if isinstance(source, dict):
        if source.get("type") == desired_type:
            matches.append(source)
        for val in source.values():
            matches.extend(collect_by_type(val, desired_type))
    elif isinstance(source, list):
        for item in source:
            matches.extend(collect_by_type(item, desired_type))
    return matches


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


def collect_code_blocks(case: Dict, use_code_agent:bool=False) -> List[str]:
    value = case['before_refactor_code']
    blocks: List[str] = []
    for i, item in enumerate(value):
        code = f"####{i+1}\n"
        if use_code_agent:
            rel_path = _module_relative_path(item['module_path'], case['settings']['src_path'])
            rel_path = str(rel_path)
            code += f"`file_path:{rel_path}`, `class_name={item['class_name']}`, `method_name={item['method_name']}`"
        else:
            code += f"\nthe related code is: \n```python\n{item['code']}\n```"
        blocks.append(code)
    return blocks

def build_prompt(case: Dict[str, Any], test_cmd:str=None, use_code_agent=False, expected_callees=[]) -> str:
    code_sections = collect_code_blocks(case, use_code_agent)
    code_blob = "\n\n".join(code_sections)
    instruction = DEFAULT_AGENT_PROMPT if use_code_agent else DEFAULT_USER_INSTRUCTION
    if case['type'] == 'DuplicatedMethod':
        if use_code_agent:
            instruction = DUPLICATED_AGENT_PROMPT
        else:
            instruction = DUPLICATED_LLM_PROMPT
    # instruction = DUPLICATED_AGENT_PROMPT if case['type'] == 'DuplicatedMethod' else DEFAULT_USER_INSTRUCTION
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        instructions=instruction.strip(),
        test=test_cmd,
        code=code_blob,
        expected_callee_position= "\n".join(expected_callees) if use_code_agent else ""
    )



def case_identifier(case: Dict[str, Any], fallback_index: int) -> str:
    for key in ("instance_id", "id", "uuid", "name", "method_name", "module_path"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"case_{fallback_index}"

def _extract_common_block(text: str) -> str:
    matches = list(COMMON_FENCE_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1)
    return None

def _extract_python_block(text: str) -> str:
    matches = list(PYTHON_FENCE_PATTERN.finditer(text))
    python_blocks = []
    for i in range(len(matches)):
        block = matches[i].group(1)
        if 'def' in block:
            python_blocks.append(block)
    # matches = list(GENERIC_FENCE_PATTERN.finditer(text))
    # if matches:
    #     return matches[-1].group(1)
    return python_blocks


def _get_source_segment(code: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(code, node)
    if segment:
        return segment
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        lines = code.splitlines()
        start = max(node.lineno - 1, 0)
        end = max(getattr(node, "end_lineno", node.lineno) - 1, start)
        return "\n".join(lines[start : end + 1])
    return ""


class _CallCollector(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: List[CallSite] = []

    def _call_name(self, node: ast.AST) -> str:
        func = getattr(node, "func", None)
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        snippet = _get_source_segment(self.source, node)
        if not snippet and hasattr(ast, "unparse"):
            try:
                snippet = ast.unparse(node)
            except Exception:
                snippet = ""
        snippet = snippet.strip()
        if snippet:
            self.calls.append(CallSite(code=snippet, name=self._call_name(node)))
        self.generic_visit(node)


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, source: str, class_name=None, method_name=None, lineno=None) -> None:
        self.source = source
        self.functions: List[FunctionSnippet] = []
        self.scope: List[str] = []
        self.class_name = class_name
        self.method_name = method_name
        self.lineno = lineno
        

    def _record_function(self, node: ast.AST, name: str) -> None:
        source = _get_source_segment(self.source, node).strip()
        collector = _CallCollector(self.source)
        collector.visit(node)
        snippet = FunctionSnippet(
            name=name,
            source=source,
            calls=collector.calls,
        )
        decorator_lines = [
            getattr(decorator, "lineno", None)
            for decorator in getattr(node, "decorator_list", [])
            if hasattr(decorator, "lineno")
        ]
        decorator_lines = [line for line in decorator_lines if line is not None]
        base_lineno = getattr(node, "lineno", None)
        start_candidates = decorator_lines + ([base_lineno] if base_lineno is not None else [])
        start_lineno = min(start_candidates) if start_candidates else base_lineno
        end_lineno = getattr(node, "end_lineno", start_lineno)
        snippet.meta["lineno"] = start_lineno
        snippet.meta["end_lineno"] = end_lineno
        snippet.meta.setdefault("class_name", self.scope[-1] if self.scope else None)
        self.functions.append(snippet)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
        if not self.class_name or (self.class_name == node.name):
            if self.is_inlineno(node):
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

    def is_inlineno(self, node):
        start_lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", start_lineno)
        if not self.lineno or not start_lineno:
            return True
        for lineno in self.lineno:
            if lineno > start_lineno and lineno < end_lineno:
                return True
        return False


    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        if not self.method_name or (self.method_name == node.name):
            if self.is_inlineno(node):
                qual_name = ".".join(self.scope + [node.name]) if self.scope else node.name
                self._record_function(node, qual_name)
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        if not self.method_name or (self.method_name == node.name):
            qual_name = ".".join(self.scope + [node.name]) if self.scope else node.name
            self._record_function(node, qual_name)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()


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

def _find_functions_from_block(code_block: str, class_name=None, method_name=None):
    if not code_block:
        return []
    if not class_name and not method_name:
        return
    code_block = normalize_snippet(code_block)
    try:
        tree = ast.parse(code_block)
    except SyntaxError:
        print("code block could not be parsed:")
        return
    collector = _FunctionCollector(code_block, class_name=class_name, method_name=method_name)
    collector.visit(tree)
    functions = collector.functions
    return functions



def parse_model_prediction(raw_text: str, case: Dict[str, Any]) -> PredictionArtifacts:
    code_blocks = _extract_python_block(raw_text)
    common_blocks = _extract_common_block(raw_text)

    common_function = _extract_functions_from_block(common_blocks, common=True)
    caller_fns: List[FunctionSnippet] = []
    duplicated_callees = set()
    functions_map:Dict[Tuple, List] = {}
    for code_block in code_blocks:
        function = _extract_functions_from_block(code_block)
        for fn in function:
            for code in case['before_refactor_code']:
                if fn.name.split(".")[-1] == code['method_name']:
                    fn.meta = {
                        "module_path": code['module_path'],
                        "class_name": code.get('class_name'),
                        "method_name": code['method_name'],
                    }
                    caller_fns.append(fn)
                    functions_map[(code['module_path'], code.get('class_name'), code['method_name'])] = function
                    break
    caller_segments: List[Segment] = []
    callee_segments: List[Segment] = []
    for caller_fn in caller_fns:
        callee_name = [c.name for c in caller_fn.calls]
        module_path, class_name, method_name = caller_fn.meta['module_path'], caller_fn.meta.get('class_name'), caller_fn.meta['method_name']
        functions = functions_map[(module_path, class_name, method_name)]
        functions.extend(common_function)
        callee_functions = [fn for fn in functions if fn.name.split(".")[-1] in callee_name]
        used_helpers: List[FunctionSnippet] = []
        calls_by_helper: Dict[str, List[str]] = {}
        caller_position = {"module_path": module_path, "class_name": class_name, "method_name": method_name}
        caller_meta: Dict[str, Any] = {"name": caller_fn.name, "position": caller_position, "type": "caller", "callees": []}
        helper_lookup: Dict[str, List[FunctionSnippet]] = {}
        for helper in callee_functions:
            key = helper.name.split(".")[-1].lower()
            helper_lookup.setdefault(key, []).append(helper)
        seen_helpers: set[str] = set()
        for call in caller_fn.calls:
            if not call.name:
                continue
            key = call.name.lower()
            for helper in helper_lookup.get(key, []):
                calls_by_helper.setdefault(helper.name, []).append(call.code)
                if helper.name not in seen_helpers:
                    used_helpers.append(helper)
                    seen_helpers.add(helper.name)
        for helper in used_helpers:
            entry = {
                "type": "callee",
                "name": helper.name,
                "code": helper.source,
            }
            caller_meta["callees"].append(entry)
            if helper.name in duplicated_callees:
                continue
            duplicated_callees.add(helper.name)
            callee_segments.append(Segment(text=helper.source, name=helper.name.split(".")[-1], meta={"type": "callee"}))
        caller_segments.append(Segment(text=code_block, meta=caller_meta, name=caller_fn.name))

    return PredictionArtifacts(
        # functions=functions,
        caller_segments=caller_segments,
        callee_segments=callee_segments,
        test_passed=None,
        raw_response=raw_text,
        parsed_payload=None,
    )


def parse_ground_truth(case: Dict[str, Any], cascade=False) -> Dict[str, List[Segment]]:
    def build_function_map(functions: List[FunctionSnippet]) -> Dict[str, List[FunctionSnippet]]:
        mapping: Dict[str, List[FunctionSnippet]] = {}
        for fn in functions:
            key = fn.name.split(".")[-1].lower()
            mapping.setdefault(key, []).append(fn)
        return mapping

    def normalize_callees(caller: Dict[str, Any], callees: List[Dict], func_map: Dict[str, List[FunctionSnippet]]) -> List[Dict[str, Any]]:
        caller_name = caller['position']['method_name']
        normal_callees: List[Dict[str, Any]] = []
        local_callees: List[Dict[str, Any]] = []
        candidate_callee_names = set()
        duplicated_methods = set()
        seen_sources: Set[str] = set()

        if caller_name in func_map:
            candidate_callee_names = {callee.name for caller in func_map[caller_name] for callee in caller.calls if callee.name in func_map}
        
        while len(callees) > 0:
            callee = callees.pop()
            key =  (callee['position']['module_path'], callee['position']['class_name'], callee['position']['method_name'])
            if key in duplicated_methods:
                continue
            duplicated_methods.add(key)
            code_text = callee.get("code", "")
            normalized_code = normalize_snippet(code_text)
            if normalized_code in seen_sources:
                continue
            seen_sources.add(normalized_code)
            callee_name = callee['position']['method_name']
            if cascade:
                callees.extend(callee.get('callees', []))
            if callee_name in candidate_callee_names:
                normalized_callee = {
                    "type": "callee",
                    "source": "local",
                    "code": func_map[callee_name][0].source,
                    "position": caller['position']
                    }
                normalized_callee['position']['method_name'] = callee_name
                local_callees.append(normalized_callee)
            else:
                normalized_callee = {
                    "type": "callee",
                    "source": "label",
                    "code": code_text,
                    "position": callee['position']
                }
                normal_callees.append(normalized_callee)
        
        # for callee in callees:
        #     key =  (callee['position']['module_path'], callee['position']['class_name'], callee['position']['method_name'])
        #     if key in duplicated_methods:
        #         continue
        #     duplicated_methods.add(key)
        #     code_text = callee.get("code", "")
        #     normalized_code = normalize_snippet(code_text)
        #     if normalized_code in seen_sources:
        #         continue
        #     seen_sources.add(normalized_code)
        #     normalized_callee = {
        #         "type": "callee",
        #         "source": "label",
        #         "code": code_text,
        #         "position": callee['position']
        #     }
        #     normal_callees.append(normalized_callee)
        return normal_callees, local_callees

    callers: List[Segment] = []
    callees: List[Segment] = []
    local_callees: List[Segment] = []
    seen_callee_segments: Set[Tuple[str, str]] = set()
    seen_local_segments: Set[Tuple[str, str]] = set()
    entries = case.get("after_refactor_code")
    for caller in entries:
        code = caller.get("code", "")
        caller_name = caller.get("position")['method_name']
        functions = _extract_functions_from_block(code)
        func_map = build_function_map(functions)
        filtered_callees, local_callee = normalize_callees(caller, caller.get("callees").copy(), func_map)
        caller_meta = {
            "type": "caller",
            "position": caller.get("position"),
            "callees": filtered_callees,
        }
        callers.append(Segment(text=code, meta=caller_meta, name=caller_name))
        for segment in convert_to_segments(filtered_callees, "callee", CODE_TEXT_FIELDS):
            key = (
                segment.text,
                json.dumps(segment.meta, ensure_ascii=False, sort_keys=True),
            )
            if key in seen_callee_segments:
                continue
            seen_callee_segments.add(key)
            callees.append(segment)
        for segment in convert_to_segments(local_callee, "local", CODE_TEXT_FIELDS):
            key = (
                segment.text,
                json.dumps(segment.meta, ensure_ascii=False, sort_keys=True),
            )
            if key in seen_local_segments:
                continue
            seen_local_segments.add(key)
            local_callees.append(segment)

    return {
        "callees": callees,
        "callers": callers,
        "local_callees": local_callees,
    }


def filter_prediction_callees(
    prediction_callees: List[Segment],
    local_callees: List[Segment],
) -> List[Segment]:
    """Remove predictions that correspond to helpers defined locally in the caller."""
    local_name_set = {
        segment.name.split(".")[-1]
        for segment in local_callees
    }
    if not local_name_set:
        return prediction_callees

    filtered: List[Segment] = []
    for segment in prediction_callees:
        name = segment.name.split(".")[-1]
        if name in local_name_set:
            continue
        filtered.append(segment)
    return filtered


def similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def match_segments(
    predictions: List[Segment],
    references: List[Segment],
    threshold: float,
    score_fn,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    used_predictinos : set[int] = set()
    for ref_idx, ref in enumerate(references):
        if ref.meta['source'] == 'local':
            continue
        best: Optional[Dict[str, Any]] = None
        for pred_idx, pred in enumerate(predictions):
            ## Only match non-local references, local means the ones defined in the caller originally.
            if pred_idx in used_predictinos:
                continue
            score = score_fn(pred.text, ref.text)
            if score < threshold:
                continue
            if best is None or score > best["score"]:
                best = {"pred_index": pred_idx, "ref_index": ref_idx, "score": score}
        if best is None:
            continue
        matches.append(best)
        used_predictinos.add(best["pred_index"])
    return matches



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
    


def codebleu_similarity(pred: str, ref: str) -> float:
    pred = strip_python_comments(pred)
    ref = strip_python_comments(ref)
    if not pred.strip() and not ref.strip():
        return 0.0
    result = calc_codebleu([pred], [ref], lang='python')
    return result['codebleu']

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




def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if isinstance(value, (int, float))]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


class RefactorEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        _input = os.path.join(args.benchmark_file)
        self.data_path = _input
        if _input.endswith(".jsonl"):
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for d in f.readlines():
                    d = json.loads(d)
                    data.append(d)
        elif _input.endswith(".json"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"unknown extension of the benchmark file {_input}")
        self.project_path = os.path.join(args.project_dir)
        self.project_root = Path(args.project_dir)
        self.cases = data
        # self.project_name = args.project_name
        # self.project_repo = self.project_root / self.project_name
        # self.cases = data['refactor_codes']
        # self.settings = data['settings']
        # self.src_path = self.settings['src_path']
        # self.test_cmd = self.settings.get('test_cmd', "")
        # self.envs = self.settings.get("envs", {})
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = args.verbose
        
        self.model = args.model
        self.llm_model = args.llm_model
        self.llm_client: Optional[Client] = None

        self.llm_client = LLMFactory.create_client(client_type=self.model, model=self.llm_model, api_key=args.api_key, base_url=args.base_url)
        self.use_code_agent = isinstance(self.llm_client, AgentClient)
        self.results: List[CaseResult] = []

    def _log(self, message: str, level=0) -> None:
        if level > 0:
            if self.verbose:
                print(message)
        else:
            print(message)

    def _extract_content(self, caller_predictions: List[Segment], case):
        caller_content = []
        replaced_contents = defaultdict(list)
        original_contents = {}
        for pred in caller_predictions:
            pred_module, pred_method = pred.meta['position']['module_path'], pred.meta['position']['method_name']
            
            for before_caller in case['before_refactor_code']:
                if before_caller['module_path'] == pred_module and before_caller['method_name'] == pred_method:
                    file_content = None
                    for bad_content in case['caller_file_content']:
                        if bad_content['module_path'] == pred_module:
                            file_content = bad_content
                            break
                    if file_content is None:
                        continue
                    start = before_caller['start']
                    end = before_caller['end']
                    first_line = file_content['code'].splitlines()[start]
                    first_line_indent = len(first_line) - len(first_line.lstrip())
                    pred_lines = pred.text.splitlines()
                    pred_line_indent = len(pred_lines[0]) - len(pred_lines[0].lstrip())
                    indent_diff = first_line_indent - pred_line_indent
                    if indent_diff > 0:
                        pred_lines = [(" " * indent_diff) + line if line.strip() else line for line in pred_lines]
                    elif indent_diff < 0:
                        pred_lines = [line[(-indent_diff):] if len(line) > (-indent_diff) else line.lstrip() for line in pred_lines]
                    replaced_contents[pred_module].append({"start": start, "end": end, "code": "\n".join(pred_lines)})
                    original_contents[pred_module] = file_content['code']
                    break
                    
        for module_path, replacements in replaced_contents.items():
            replacements.sort(key=lambda x: x['start'], reverse=True)
            original_content = original_contents[module_path].splitlines() 
            for replacement in replacements:
                start = replacement['start']
                end = replacement['end']
                original_content[start:end] = replacement['code'].splitlines()
            caller_content.append({
                "code": "\n".join(original_content),
                "module_path": module_path,
            })
            break
        return caller_content

    


    def _write_ground_truth_files(self, case: Dict[str, Any]) -> List[str]:
        written: List[str] = []
        project_repo = self.get_project_path(case)
        for file_entry in case.get("caller_file_content", []):
            module_path = file_entry.get("module_path")
            if not module_path:
                continue
            rel_path = _module_relative_path(module_path, case['settings']['src_path'])
            abs_path = project_repo / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(file_entry.get("code", ""), encoding="utf-8")
            written.append(str(rel_path).replace("\\", "/"))
        return written

    def _has_staged_changes(self, case) -> bool:
        project_repo = self.get_project_path(case)
        result = _run_git_command(["diff", "--cached", "--quiet"], check=False, cwd=project_repo)
        if result.returncode in (0, 1):
            return result.returncode == 1
        raise RuntimeError(f"Unexpected git diff --cached return code {result.returncode}")

    def get_project_path(self, case):
        return os.path.join(self.project_path, case['name'])

    def _read_cache_code_agent(self, case,
                               prompt_hash:str):
        cache_dir = Path("cache")
        project_name = case['name']
        cache_dir = cache_dir / project_name

        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"code_agent_{self.model}_{self.llm_client.model}_{project_name}.json"

        if not cache_path.exists():
            return None, None
        cache_json = json.loads(cache_path.read_text(encoding="utf-8"))
        if prompt_hash not in cache_json:
            return None, None
        payload = cache_json[prompt_hash]
        diff_text = payload['diff']
        diff_files = payload['diff_files']
        output_text = payload.get("output_text", "")
        stat = payload.get("stat", None)
        if stat:
            response:AgentResponse= AgentResponse(content=output_text, model=self.llm_client.model, **stat)
            response.raw_response = payload.get("raw_response", None)
        tmp_file = cache_dir / f"tmp_{str(uuid.uuid4())}.diff"
        tmp_file.write_text(diff_text)
        try:
            _run_git_command(['apply', str(tmp_file.absolute())], cwd=self.get_project_path(case))
        except:
            self._log("read agent cache failed")
            return None, None
        finally:
            ## delete temp files
            tmp_file.unlink()
        return 1, (output_text, response, diff_files, diff_text)

        

    def _cache_code_agent_diff(
        self,
        case,
        prompt_hash: str,
        response: LLMResponse,
        diff_text: str,
        diff_files: List[str],
    ) -> None:
        if len(diff_files) == 0:
            return
        instance_id = case['instance_id']
        project_name = case['name']
        cache_dir = Path("cache")
        cache_dir = cache_dir / project_name
        cache_dir.mkdir(exist_ok=True)
        agent_model = self.llm_client.model
        stat = self.unpack_response(response)
        payload = {"instance_id": instance_id, "raw_response": response.raw_response, "prompt_hash": prompt_hash,"model": agent_model,"agent": self.model,"project_name": project_name,"output_text": response.content,"stat": stat,"diff_files": diff_files,"diff": diff_text}
        cache_path = cache_dir / f"code_agent_{self.model}_{agent_model}_{project_name}.json"
        if cache_path.exists():
            cache_json = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            cache_json = {}
        cache_json[prompt_hash] = payload
        cache_path.write_text(json.dumps(cache_json, ensure_ascii=False, indent=2), encoding="utf-8")

    def _modules_from_paths(self, case, diff_files: List[str]) -> Set[str]:
        modules: Set[str] = set()
        for diff_file in diff_files:
            module_name = self._module_from_diff_path(case, diff_file)
            if module_name:
                modules.add(module_name)
        return modules

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
        if PatchSet is None:
            return self._parse_diff_changed_lines_manual(case, diff_text)
        changed_lines: Dict[str, Set[int]] = defaultdict(set)
        try:
            patch = PatchSet(diff_text, encoding="utf-8")
        except Exception:
            return self._parse_diff_changed_lines_manual(case, diff_text)
        for patched_file in patch:
            target_path = getattr(patched_file, "path", None) or patched_file.target_file
            if not target_path or target_path == "/dev/null":
                continue
            if target_path.startswith("b/"):
                target_path = target_path[2:]
            module_path = self._module_from_diff_path(case, target_path)
            if not module_path:
                continue
            module_lines = changed_lines[module_path]
            for hunk in patched_file:
                new_line = hunk.target_start
                if new_line is None:
                    continue
                for line in hunk:
                    if getattr(line, "is_added", False):
                        module_lines.add(new_line)
                        new_line += 1
                    elif getattr(line, "is_context", False):
                        new_line += 1
                    elif getattr(line, "is_removed", False):
                        module_lines.add(new_line)
                    else:
                        new_line += 1
        return changed_lines

    def _function_overlaps_changed_lines(
        self,
        fn: FunctionSnippet,
        changed_lines: Optional[Set[int]],
    ) -> bool:
        if changed_lines is None:
            return True
        if not changed_lines:
            return False
        start = fn.meta.get("lineno")
        end = fn.meta.get("end_lineno", start)
        if start is None or end is None:
            return True
        for line_no in range(start, end + 1):
            if line_no in changed_lines:
                return True
        return False

    def _filter_functions_by_changed_lines(
        self,
        functions: List[FunctionSnippet],
        changed_lines: Optional[Set[int]],
    ) -> List[FunctionSnippet]:
        if changed_lines is None:
            return functions
        if not changed_lines:
            return []
        return [fn for fn in functions if self._function_overlaps_changed_lines(fn, changed_lines)]

    def _collect_label_callees(self, case: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen: Set[Tuple[str, Optional[str], str]] = set()
        for caller in case.get("after_refactor_code", []):
            callees = caller.get("callees", []).copy()
            while len(callees) > 0:
                callee = callees.pop()
                position = callee.get("position") or {}
                module_path = position.get("module_path")
                method_name = position.get("method_name")
                class_name = position.get("class_name")
                if not module_path or not method_name:
                    continue
                # ignore the callee that recursive calling
                if caller['position'] == position:
                    continue
                key = (module_path, class_name, method_name)
                if key in seen:
                    continue
                seen.add(key)
                callees.extend(callee.get('callees', []))
                grouped[module_path].append(
                    {
                        "class_name": class_name,
                        "method_name": method_name,
                    }
                )
        return grouped

    def _find_function_spans(
        self,
        source: str,
        targets: Set[Tuple[Optional[str], str]],
    ) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        if not targets:
            return spans
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return spans

        class _SpanCollector(ast.NodeVisitor):
            def __init__(self, target_lookup: Set[Tuple[Optional[str], str]]) -> None:
                self.targets = target_lookup
                self.class_stack: List[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def _record_span(self, node: ast.AST, name: str) -> None:
                class_name = self.class_stack[-1] if self.class_stack else None
                key = (class_name, name)
                lineno = getattr(node, "lineno", None)
                end_lineno = getattr(node, "end_lineno", None)
                if key not in self.targets or lineno is None or end_lineno is None:
                    return
                decorators = getattr(node, "decorator_list", [])
                decorator_lines = [
                    getattr(decorator, "lineno", lineno) for decorator in decorators if hasattr(decorator, "lineno")
                ]
                if decorator_lines:
                    lineno = min(lineno, min(decorator_lines))
                spans.append(
                    {
                        "class_name": class_name,
                        "method_name": name,
                        "start": lineno,
                        "end": end_lineno,
                    }
                )

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
                self._record_span(node, node.name)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
                self._record_span(node, node.name)
                self.generic_visit(node)

        collector = _SpanCollector(targets)
        collector.visit(tree)
        return spans

    def _remove_ground_truth_callees(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        callee_map = self._collect_label_callees(case)
        project_repo = self.get_project_path(case)
        removal_records: List[Dict[str, Any]] = []
        src_path = case['settings']['src_path']
        for module_path, entries in callee_map.items():
            rel_path = _module_relative_path(module_path, src_path)
            abs_path = project_repo / rel_path
            if not abs_path.exists():
                continue
            try:
                source = abs_path.read_text(encoding="utf-8")
            except OSError:
                continue
            targets = {(entry["class_name"], entry["method_name"]) for entry in entries}
            spans = self._find_function_spans(source, targets)
            if not spans:
                continue
            lines = source.splitlines(keepends=True)
            file_records: List[Dict[str, Any]] = []
            for span in sorted(spans, key=lambda item: item["start"], reverse=True):
                start_idx = max(span["start"] - 1, 0)
                end_idx = min(span["end"], len(lines))
                if start_idx >= len(lines):
                    continue
                original_segment = "".join(lines[start_idx:end_idx])
                if not original_segment.strip():
                    continue
                placeholder_token = f"__PLACEHOLDER__{uuid.uuid4().hex}__"
                placeholder_line = f"# {placeholder_token}\n"
                lines[start_idx:end_idx] = [placeholder_line]
                file_records.append(
                    {
                        "file_path": abs_path,
                        "placeholder": placeholder_line,
                        "original": original_segment,
                        "module_path": module_path,
                        "class_name": span.get("class_name"),
                        "method_name": span["method_name"],
                    }
                )
            if file_records:
                abs_path.write_text("".join(lines), encoding="utf-8")
                removal_records.extend(file_records)
        return removal_records

    def _restore_ground_truth_callees(self, case, removals: List[Dict[str, Any]]) -> None:
        if not removals:
            return
        project_repo = self.get_project_path(case)
        grouped: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
        src_path = case['settings']['src_path']
        for entry in removals:
            grouped[entry["file_path"]].append(entry)
        for path, entries in grouped.items():
            if not path.exists():
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            changed = False
            for entry in entries:
                placeholder = entry["placeholder"]
                original = entry["original"]
                module_path = entry.get("module_path")
                class_name = entry.get("class_name")
                method_name = entry.get("method_name")
                if placeholder not in source:
                    continue
                if module_path:
                    rel_path = _module_relative_path(module_path, src_path)
                    expected_path = project_repo / rel_path
                    if expected_path.resolve() != path.resolve():
                        continue
                new_functions = _find_functions_from_block(source,class_name=class_name, method_name=method_name)
                match = None
                for fn in new_functions:
                    components = fn.name.split(".")
                    simple_name = components[-1]
                    candidate_class = components[-2] if len(components) > 1 else None
                    if simple_name != method_name:
                        continue
                    if class_name and candidate_class != class_name:
                        continue
                    match = fn
                    break
                ## if find same method name and class name functions, then skip it to avoid duplicated name
                if match:
                    continue
                source = source.replace(placeholder, original, 1)
                changed = True
            if changed:
                path.write_text(source, encoding="utf-8")

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

    def _build_prediction_from_repo(
        self,
        case: Dict[str, Any],
        diff_files: List[str],
        diff_text: str,
        response: LLMResponse,
    ) -> PredictionArtifacts:
        project_repo = self.get_project_path(case)
        changed_modules = self._modules_from_paths(case, diff_files)
        changed_line_map = self._parse_diff_changed_lines(case, diff_text) if diff_text else {}
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
                module_path = self._module_from_diff_path(case, diff_file)
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
                match = self._match_function(entry, active_functions)
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
            response=self.unpack_response(response),
            raw_response=diff_text,
            parsed_payload={
                "backend": "code_agent",
                "changed_modules": sorted(changed_modules),
            },
        )
    def unpack_response(self, response: LLMResponse):
        return {k: v for k, v in vars(response).items() if isinstance(v, int) or isinstance(v, float)}
    def _run_code_agent_workflow(
        self,
        instance_id: str,
        case: Dict[str, Any],
        prompt: str,
        prompt_hash: str,
    ) -> Tuple[PredictionArtifacts, bool]:
        original_head = case['commit_hash']
        diff_text = ""
        diff_files: List[str] = []
        success = False
        prediction = None
        project_name = case['name']
        project_repo = self.get_project_path(case)
        try:
            self._log("committing bad code")
            _run_git_command(["reset", "--hard", original_head], cwd=project_repo)
            written_files = self._write_ground_truth_files(case)
            intermit_commit_id = None
            if written_files:
                _run_git_command(["add", *written_files], cwd=project_repo)
                if self._has_staged_changes(case):
                    _run_git_command(["commit", "-m", f"[baseline] {instance_id}"], cwd=project_repo)
                    intermit_commit_id = _run_git_command(["rev-parse", "HEAD"], cwd=project_repo).stdout
                    intermit_commit_id = intermit_commit_id.strip()
            
            is_cached, cached_info = self._read_cache_code_agent(case, prompt_hash)
            if (not self.args.force_request) and is_cached:
                self._log("reading cache")
                output_text, response, diff_files, diff_text = cached_info
                if output_text:
                    self._log(output_text, 1)
            else:
                # return None, False
                self._log("use code agent")
                # Hide reference callees before invoking the agent to avoid data leakage.
                removal_records = self._remove_ground_truth_callees(case)
                edit_files = [os.path.relpath(r['file_path'], project_repo) for r in removal_records]
                _run_git_command(["add", *edit_files], cwd=project_repo)
                
                if self._has_staged_changes(case):
                    _run_git_command(["commit", "-m", f"[remove] {instance_id}"], cwd=project_repo)
                    
                # with disableGitTools(project_repo):
                ex = None
                try:
                    raise Exception("hhhh")
                    response = self.llm_client.chat(prompt, project_repo=project_repo)
                    invoke_success = response is not None
                    self._log(response.content)
                except Exception as exp:
                    print(exp)
                    ex = exp
                    invoke_success = False
                if not invoke_success:
                    return ex, False
                # self._restore_ground_truth_callees(case, removal_records)
                if intermit_commit_id:
                    diff_text = _run_git_command(["diff", intermit_commit_id], cwd=project_repo).stdout
                    diff_output = _run_git_command(["diff", "--name-only", intermit_commit_id], cwd=project_repo).stdout
                else:
                    diff_text = _run_git_command(["diff"], cwd=project_repo).stdout
                    diff_output = _run_git_command(["diff", "--name-only"], cwd=project_repo).stdout
                diff_files = [line.strip() for line in diff_output.splitlines() if line.strip()]
                self._cache_code_agent_diff(case, prompt_hash, response, diff_text, diff_files)
            statis = self.unpack_response(response)
            self._log(str(statis))
            self._log("parsing predictions")
            prediction = self._build_prediction_from_repo(case, diff_files, diff_text, response)
            self._log("running testunit")
            success, output = run_project_tests(project_name, project_repo, case["testsuites"], envs=case['settings']['envs'], test_cmd=case['settings']['test_cmd'], timeout=60)
            self._log(f"testunit {'pass' if success else 'fail'}")
            
        finally:
            _run_git_command(["reset", "--hard", original_head], cwd=project_repo)
        return prediction, success

    def cache_result(self, case, caseResult:CaseResult|Dict):
        if caseResult is None:
            return
        file = Path("cache") / f"{self.model}_{self.llm_model}_result.json"
        ret = {}
        if file.exists():
            with open(file) as f:
                ret = json.load(f)
        if isinstance(caseResult, Dict):
            ret[case['instance_id']] = caseResult
        else:
            ret[case['instance_id']] = {k: v for k, v in vars(caseResult).items()}

        with open(file, "w") as f:
            f.write(json.dumps(ret))
    def read_cache_result(self, case):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        file = Path("cache") / f"{self.model}_{self.llm_model}_result.json"
        instance_id = case['instance_id']
        if file.exists():
            with open(file) as f:
                ret = json.load(f)
                if instance_id in ret:
                    if "exception" in ret[instance_id]:
                        return ret
                    caseResult = CaseResult(**ret[instance_id])
                    return caseResult
                
    def require_lock(self, project_name):
        locks = []
        if os.path.exists(".repo_lock"):
            with open(".repo_lock") as f:
                locks = [l.strip() for l in f.readlines()]
        if project_name in locks:
            return False
        locks.append(project_name)
        with open(".repo_lock", "w") as f:
            f.write("\n".join(locks))
        return True
    def unlock(self, project_name):
        locks = []
        if os.path.exists(".repo_lock"):
            with open(".repo_lock") as f:
                locks = [l.strip() for l in f.readlines()]
        locks = [x for x in locks if x != project_name]
        with open(".repo_lock", "w") as f:
            f.write("\n".join(locks))

    def run(self) -> Dict[str, Any]:
        total_cases = len(self.cases)
        self._log(f"Loaded {total_cases} cases from {self.data_path}")
        
        cases = []
        lines = self.cases.copy()
        sleep_interval = 200
        acc_step = 0
        while len(lines) > 0:
            case = lines.pop(0)
            project_name = case['name']
            if self.args and self.args.project_name is not None:
                if self.args.project_name != project_name:
                    continue
            instance_id = case['instance_id']
            result = self.read_cache_result(case)
            if self.args.force_request or result is None:
                if not self.require_lock(project_name):
                    self._log("do not get the lock of " + project_name)
                    lines.append(case)
                    acc_step += 1
                    if acc_step >= sleep_interval:
                        print(f"waiting for the lock {project_name}")
                        acc_step = 0
                        time.sleep(120)
                    continue
                self._log(f"Processing {instance_id}")
                try:
                    result = self._process_case(instance_id, case)
                    if isinstance(result, Exception):
                        result = {"instance_id": case['instance_id'], "exception": str(result.__class__)}
                    self.cache_result(case, result)
                    
                except Exception as e:
                    self._log(e)
                    result = None
                finally:
                    self.unlock(project_name)
            if result is None:
                continue
            self.results.append(result)
            cases.append(case)
            # if isinstance(result, Dict):
            #     break
            
        per_case_path = self.output_dir / f"{self.args.model}_{self.args.llm_model}_per_case_results.json"
        serialized = [dataclasses.asdict(result) if isinstance(result, CaseResult) else result for case, result in zip(cases, self.results)]
        per_case_path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = self._summarize(cases)
        summary_path = self.output_dir / f"{self.args.model}_{self.args.llm_model}_evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _process_case(self, instance_id: str, case: Dict[str, Any]) -> CaseResult:
        # if case['type'] == 'LongMethod':
        #     return
        # if case['meta']['depth'] == 1:
        #     return
        project_name = case['name']
        spec = get_spec(project_name)
        project_repo = self.get_project_path(case)
        src_path = case['settings']['src_path']
        if not prepare_to_run(spec):
            return
        ground_truth = parse_ground_truth(case, cascade=True)
        callees = [f"file_path={str(_module_relative_path(c.meta['position']['module_path'], src_path))}, class_name={c.meta['position']['class_name']}, method_name={c.meta['position']['method_name']}" for c in ground_truth['callees']] 
        test_cmd = create_test_command(test_file_paths=case['testsuites'], test_cmd=case['settings']['test_cmd'], envs=case['settings']['envs'], use_envs=True) if self.args.use_test else DEFAULT_FORBID_TEST
        prompt = build_prompt(case, use_code_agent=self.use_code_agent, test_cmd=test_cmd, expected_callees=callees)
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        original_head = case['commit_hash']
        _run_git_command(["reset", "--hard", original_head], cwd=project_repo)
        if self.use_code_agent:
            prediction, success = self._run_code_agent_workflow(instance_id, case, prompt, prompt_hash)
            if prediction is None:
                return None
            elif isinstance(prediction, Exception):
                return prediction
        else:
            payload = self._predict(prompt, prompt_hash, instance_id, case)
            prediction = parse_model_prediction(payload["response_text"], case)
            caller_content = self._extract_content(prediction.caller_segments, case)
            success = replace_and_test_caller(
                project_name=project_name,
                src_path=src_path,
                testsuites=case['testsuites'],
                caller_file_content=caller_content,
                project_dir=self.project_path,
                commit_hash=case['commit_hash'],
                envs=case['settings']['envs'],
                test_cmd=case['settings']['test_cmd']
            )
        filtered_prediction_callees = filter_prediction_callees(prediction.callee_segments, ground_truth['local_callees'])
        callee_matches = match_segments(
            filtered_prediction_callees,
            ground_truth["callees"],
            self.args.similarity_threshold,
            codebleu_similarity,
        )
        callee_precision, callee_recall, callee_f1 = compute_prf(
            len(callee_matches),
            len(filtered_prediction_callees),
            len(ground_truth["callees"]),
        )
        caller_predictions = prediction.caller_segments

        if len(prediction.caller_segments) == 0:
            
            self._log(f"No caller predictions found for case {instance_id}.")
            return CaseResult(
                instance_id=instance_id,
                prompt_hash=prompt_hash,
                caller_accuracy=0,
                callee_precision=callee_precision,
                callee_recall=callee_recall,
                callee_f1=callee_f1,
                match_scores=0.0,
                response_stat=prediction.response if prediction is not None else None,
                test_passed=success
            )

        if len(callee_matches) == 0:
            match_scores = 0.0
        else:
            match_scores = sum([round(item["score"], 3) for item in callee_matches]) / len(callee_matches)
        details = {
            "num_predicted_callees": len(filtered_prediction_callees),
            "num_target_callees": len(ground_truth["callees"]),
            "num_predicted_calls": len(caller_predictions),
            "callee_match_scores": match_scores,
            "prediction_backend": "code_agent" if self.use_code_agent else "llm",
        }
        self._log(f"case {instance_id} result: callee_f1:{callee_f1}, callee_match: {callee_matches}, testunit pass:{success} ")
        return CaseResult(
            instance_id=instance_id,
            prompt_hash=prompt_hash,
            caller_accuracy= len(prediction.caller_segments) / len(ground_truth['callers']) if len(ground_truth['callers']) > 0 else 0,
            callee_precision=callee_precision,
            callee_recall=callee_recall,
            callee_f1=callee_f1,
            callee_match_score=match_scores,
            test_passed=success,
            response_stat=prediction.response if prediction is not None else None,
            details=details,
        )

    def _predict(
        self,
        prompt: str,
        prompt_hash: str,
        instance_id: str,
        case: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            raise RuntimeError(f"Cache miss for {instance_id} and no LLM client available.")
        response_text = self.llm_client.chat(
            prompt=prompt,
            temperature=self.args.temperature,
        )
        payload = {
            "prompt_hash": prompt_hash,
            "model": self.args.model,
            "instance_id": instance_id,
            "prompt": prompt,
            "response_text": response_text,
            "metadata": {
                "project_name": case.get("project_name"),
                "module_path": case.get("module_path"),
            },
        }
        return payload

    def _summarize(self, cases) -> Dict[str, Any]:
        
        def summary_result(results):
            
            summary = {
                "cases_evaluated": len(results),
                "output_dir": str(self.output_dir),
            }
            summary["averages"] = {
                "caller_accuracy": mean_or_none(result.caller_accuracy for result in results if isinstance(result, CaseResult)),
                "callee_match_score": mean_or_none(result.callee_match_score for result in results if isinstance(result, CaseResult)),
                "callee_precision": mean_or_none(result.callee_precision for result in results if isinstance(result, CaseResult)),
                "callee_recall": mean_or_none(result.callee_recall for result in results if isinstance(result, CaseResult)),
                "callee_f1": mean_or_none(result.callee_f1 for result in results if isinstance(result, CaseResult)),
            }
                
            test_values = [result.test_passed for result in results if isinstance(result, CaseResult) and result.test_passed is not None]
            if test_values:
                summary["test_pass_rate"] = sum(1 for value in test_values if value) / len(test_values)
            else:
                summary["test_pass_rate"] = 0
            if len(results) > 0:
                summary['exec_success_rate'] = len([r for r in results if isinstance(r, CaseResult)]) / len(results)
            else:
                summary['exec_success_rate'] = 0
            return summary
        summary = {}
        summary['total'] = summary_result(self.results)
        depth_1 = [ret for ret, case in zip(self.results, cases) if ('depth' in case['meta'] and case['meta']['depth'] == 1)]
        summary['depth_1'] = summary_result(depth_1)
        depth_2 = [ret for ret, case in zip(self.results, cases) if ('depth' in case['meta'] and case['meta']['depth'] == 2)]
        summary['depth_2'] = summary_result(depth_2)
        depth_3 = [ret for ret, case in zip(self.results, cases) if ('depth' in case['meta'] and case['meta']['depth'] == 3)]
        summary['depth_3'] = summary_result(depth_3)
        long_summary = [ret for ret, case in zip(self.results, cases) if (case['type'] == 'Long')]
        summary['long'] = summary_result(long_summary)
        duplicated_summary = [ret for ret, case in zip(self.results, cases) if (case['type'] == 'Duplicated')]
        summary['duplicated'] = summary_result(duplicated_summary)
        datasets = set( [case['name'] for case in cases])
        for d in datasets:
            dataset_summary = summary_result([ret for ret, case in zip(self.results, cases) if (case['name'] == d)])
            summary[d] = dataset_summary
        return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM refactor ability against reference data.")
    parser.add_argument("--output-dir", default="run/refactor_eval", help="Directory for cached outputs and reports.")
    parser.add_argument("--model", default="claude_code", help="Model name used for generation.")
    parser.add_argument("--llm_model", default=None, nargs="+", help="the LLM model")
    parser.add_argument("--project-dir", default="../project", help="Project directory for resolving relative paths in test commands.")
    parser.add_argument("--project-name", default=None, help="Project name")
    parser.add_argument("--benchmark_file", default="output/benchmark.jsonl", type=str)
    parser.add_argument("--use-test", default=False, type=bool, help="instruct model whether to use unittest to fix errors")
    parser.add_argument("--api_key", default=None, type=str, help="will overwrite the config in .env, e.g., QWEN_CODE_API_KEY for qwen_code")
    parser.add_argument("--base_url", default=None, type=str, help="will overwrite the config in .env, e.g., QWEN_CODE_BASE_URL for qwen_code")
    parser.add_argument("--force_request", default=False, action="store_true")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature for the chat model.")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Similarity threshold used for F1 matching.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress information.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)
    # args.use_code_agent = True
    llm_models = args.llm_model
    for llm_model in llm_models:
        args.llm_model = llm_model
        evaluator = RefactorEvaluator(args)
        summary = evaluator.run()
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
