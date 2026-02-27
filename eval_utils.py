import re
import keyword
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import dataclasses
import textwrap
from prompts import *
import json
from utils import _log, hashcode, strip_python_comments, _module_relative_path,create_test_command
import ast

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

def build_prompt(case: Dict[str, Any], use_code_agent=False, use_test=False) -> str:
    code_sections = collect_code_blocks(case, use_code_agent)
    src_path = case['settings']['src_path']
    ground_truth = parse_ground_truth(case, cascade=True)
    expected_callees = [f"file_path={str(_module_relative_path(c.meta['position']['module_path'], src_path))}, class_name={c.meta['position']['class_name']}, method_name={c.meta['position']['method_name']}" for c in ground_truth['callees']] 
        
    test_cmd = create_test_command(test_file_paths=case['testsuites'], test_cmd=case['settings']['test_cmd'], envs=case['settings']['envs'], use_envs=True) if use_test else DEFAULT_FORBID_TEST
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