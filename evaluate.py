import argparse
import ast
import dataclasses
from collections import Counter
import hashlib
import json
import keyword
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from client import LLMFactory, LLMClient
from collections import defaultdict
from testunits import replace_and_test_caller


CODE_TEXT_FIELDS = ["code", "source", "body", "text", "snippet"]
CALLSITE_TEXT_FIELDS = ["code", "call_site", "call", "snippet", "text", "replacement", "details"]

DEFAULT_USER_INSTRUCTION = textwrap.dedent(
    """
    You are given one or more Python code snippets that include at least one caller method that can be refactored.
    Identify helper methods that should be extracted, output the helper implementations, and update the caller.

    Think step by step and provide the thinking flow, and finally share the result inside a single ```python ... ``` block that includes the
    refactored code.
    Note you can't change the behavior of the original code, such as adding new class including caller method which breaks the original functionality.
    """
).strip()
DUPLICATED_AGENT_PROMPT = textwrap.dedent(
    """
    You are given two Python code snippets extracted from different caller methods in the same repository. 
These snippets contain duplicated or highly similar logic that should be refactored.

Your task is to:
1. Identify the duplicated code shared across the given caller methods.
2. Extract this duplicated logic into a single reusable helper function.
3. Search the current repository for other methods that contain similar duplicated logic and refactor them to reuse the same helper.
4. Move the extracted helper function to an appropriate existing file or module in the repository (do not introduce unnecessary new files or classes).
5. Update all affected caller methods to use the extracted helper function.

Constraints:
- You must not change the original program behavior.
- Do not introduce new classes that contain the original caller methods.
- The refactoring should preserve the original control flow, inputs, and outputs.
- The helper function should be general enough to support all refactored callers.

Carefully reason about code structure, duplication patterns, and file placement.

You can do the replacements and summary the replacements in the Response block
"""
).strip()

SYSTEM_PROMPT = (
    "You are an expert Python refactoring assistant. "
)

PROMPT_TEMPLATE = """{system}

{instructions}

### Code Snippets
{code}

### Response
"""

PYTHON_FENCE_PATTERN = re.compile(r"```python[ \t]*\n([\s\S]*?)```", re.IGNORECASE)
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
    functions: List[FunctionSnippet]
    caller_segments: List[Segment]
    callee_segments: List[Segment]
    test_passed: Optional[bool]
    raw_response: str
    parsed_payload: Optional[Dict[str, Any]]
    error: Optional[str] = None


@dataclasses.dataclass
class CaseResult:
    case_id: str
    prompt_hash: str
    callee_precision: Optional[float]
    callee_recall: Optional[float]
    callee_f1: Optional[float]
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


def collect_code_blocks(case: Dict) -> List[str]:
    value = case['before_refactor_code']
    blocks: List[str] = []
    for item in value:
        blocks.append(f"`module_path:{item['module_path']}`, `class_name={item['class_name']}`, `method_name={item['method_name']}` and the related code is: \n```python\n{item['code']}\n```")
    return blocks


def build_prompt(case: Dict[str, Any]) -> str:
    code_sections = collect_code_blocks(case)
    code_blob = "\n\n".join(code_sections)
    instruction = DUPLICATED_AGENT_PROMPT if case['type'] == 'DuplicatedMethod' else DEFAULT_USER_INSTRUCTION
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        instructions=instruction.strip(),
        code=code_blob,
    )



def case_identifier(case: Dict[str, Any], fallback_index: int) -> str:
    for key in ("case_id", "id", "uuid", "name", "method_name", "module_path"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"case_{fallback_index}"


def _extract_last_python_block(text: str) -> str:
    matches = list(PYTHON_FENCE_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1)
    matches = list(GENERIC_FENCE_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1)
    return text


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
    def __init__(self, source: str) -> None:
        self.source = source
        self.functions: List[FunctionSnippet] = []
        self.scope: List[str] = []

    def _record_function(self, node: ast.AST, name: str) -> None:
        source = _get_source_segment(self.source, node).strip()
        collector = _CallCollector(self.source)
        collector.visit(node)
        self.functions.append(
            FunctionSnippet(
                name=name,
                source=source,
                calls=collector.calls,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        qual_name = ".".join(self.scope + [node.name]) if self.scope else node.name
        self._record_function(node, qual_name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        qual_name = ".".join(self.scope + [node.name]) if self.scope else node.name
        self._record_function(node, qual_name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _extract_functions_from_block(code_block: str) -> List[FunctionSnippet]:
    code_block = normalize_snippet(code_block)
    try:
        tree = ast.parse(code_block)
    except SyntaxError:
        print("code block could not be parsed:")
        return []
    collector = _FunctionCollector(code_block)
    collector.visit(tree)
    return collector.functions





def parse_model_prediction(raw_text: str, case: Dict[str, Any]) -> PredictionArtifacts:
    code_block = _extract_last_python_block(raw_text)
    functions = _extract_functions_from_block(code_block)
    caller_fns: List[FunctionSnippet] = []
    duplicated_callees = set()
    for fn in functions:
        for code in case['before_refactor_code']:
            if fn.name.split(".")[-1] == code['method_name']:
                fn.meta = {
                    "module_path": code['module_path'],
                    "class_name": code.get('class_name'),
                    "method_name": code['method_name'],
                }
                caller_fns.append(fn)
                break
    caller_segments: List[Segment] = []
    callee_segments: List[Segment] = []
    for caller_fn in caller_fns:
        callee_name = [c.name for c in caller_fn.calls]
        callee_functions = [fn for fn in functions if fn.name.split(".")[-1] in callee_name]
        
        

        used_helpers: List[FunctionSnippet] = []
        calls_by_helper: Dict[str, List[str]] = {}
        caller_position = {"module_path": caller_fn.meta['module_path'], "class_name": caller_fn.meta.get('class_name'), "method_name": caller_fn.meta['method_name']}
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
        functions=functions,
        caller_segments=caller_segments,
        callee_segments=callee_segments,
        test_passed=None,
        raw_response=raw_text,
        parsed_payload=None,
    )


def parse_ground_truth(case: Dict[str, Any]) -> Dict[str, List[Segment]]:
    def build_function_map(functions: List[FunctionSnippet]) -> Dict[str, List[FunctionSnippet]]:
        mapping: Dict[str, List[FunctionSnippet]] = {}
        for fn in functions:
            key = fn.name.split(".")[-1].lower()
            mapping.setdefault(key, []).append(fn)
        return mapping

    def normalize_callees(caller: Dict[str, Any], callees: Any, func_map: Dict[str, List[FunctionSnippet]]) -> List[Dict[str, Any]]:
        caller_name = caller['position']['method_name']
        normal_callees: List[Dict[str, Any]] = []
        local_callees: List[Dict[str, Any]] = []
        candidate_callee_names = set()
        if caller_name in func_map:
            candidate_callee_names = {callee.name for caller in func_map[caller_name] for callee in caller.calls if callee.name in func_map}
        for callee_name in candidate_callee_names:
            normalized_callee = {
                "type": "callee",
                "source": "local",
                "code": func_map[callee_name][0].source,
                "position": caller['position']
            }
            normalized_callee['position']['method_name'] = callee_name
            local_callees.append(normalized_callee)
        duplicated_methods = set()
        for callee in callees:
            key =  (callee['position']['module_path'], callee['position']['class_name'], callee['position']['method_name'])
            if key in duplicated_methods:
                continue
            duplicated_methods.add(key)
            normalized_callee = {
                "type": "callee",
                "source": "label",
                "code": callee.get("code", ""),
                "position": callee['position']
            }
            normal_callees.append(normalized_callee)
        return normal_callees, local_callees

    callers: List[Segment] = []
    callees: List[Segment] = []
    local_callees: List[Segment] = []
    entries = case.get("after_refactor_code")
    for caller in entries:
        code = caller.get("code", "")
        caller_name = caller.get("position")['method_name']
        functions = _extract_functions_from_block(code)
        func_map = build_function_map(functions)
        filtered_callees, local_callee = normalize_callees(caller, caller.get("callees"), func_map)
        caller_meta = {
            "type": "caller",
            "position": caller.get("position"),
            "callees": filtered_callees,
        }
        callers.append(Segment(text=code, meta=caller_meta, name=caller_name))
        callees.extend(convert_to_segments(filtered_callees, "callee", CODE_TEXT_FIELDS))
        local_callees.extend(convert_to_segments(local_callee, "local", CODE_TEXT_FIELDS))
    
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


def _tokenize_code(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def _ngram_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0
    pred_counts = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_counts = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum(min(count, ref_counts[ngram]) for ngram, count in pred_counts.items())
    total = sum(pred_counts.values())
    return overlap / total if total else 0.0


def _weighted_ngram_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    def build(tokens: List[str]) -> Counter:
        counts: Counter = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            weight = sum(2.0 if token in PYTHON_KEYWORDS else 1.0 for token in ngram) / n
            counts[ngram] += weight
        return counts

    pred_counts = build(pred_tokens)
    ref_counts = build(ref_tokens)
    overlap = sum(min(weight, ref_counts.get(ngram, 0.0)) for ngram, weight in pred_counts.items())
    total = sum(pred_counts.values())
    return overlap / total if total else 0.0


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    if pred_len == 0:
        return 0.0
    if pred_len > ref_len:
        return 1.0
    if ref_len == 0:
        return 0.0
    return math.exp(1 - ref_len / pred_len)


def _syntax_vector(code: str) -> Counter:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()
    counter: Counter = Counter()
    for node in ast.walk(tree):
        counter[type(node).__name__] += 1
    return counter


def _vector_similarity(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    overlap = sum(min(left[key], right.get(key, 0)) for key in left)
    total = sum(left.values()) + sum(right.values())
    return (2 * overlap / total) if total else 0.0


class _NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # type: ignore[override]
        self.names.add(node.attr)
        self.generic_visit(node)


def _dataflow_features(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    collector = _NameCollector()
    collector.visit(tree)
    return collector.names


def _set_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    precision = intersection / len(left) if left else 0.0
    recall = intersection / len(right) if right else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def codebleu_similarity(pred: str, ref: str) -> float:
    if not pred.strip() and not ref.strip():
        return 1.0
    pred_tokens = _tokenize_code(pred)
    ref_tokens = _tokenize_code(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, 5):
        prec = _ngram_precision(pred_tokens, ref_tokens, n)
        precisions.append(max(prec, 1e-9))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))
    ngram_score = bp * geo_mean

    weighted_scores = []
    for n in range(1, 5):
        weighted_scores.append(_weighted_ngram_precision(pred_tokens, ref_tokens, n))
    weighted_score = sum(weighted_scores) / len(weighted_scores)

    syntax_score = _vector_similarity(_syntax_vector(pred), _syntax_vector(ref))
    dataflow_score = _set_similarity(_dataflow_features(pred), _dataflow_features(ref))
    return (ngram_score + weighted_score + syntax_score + dataflow_score) / 4


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
        _input = os.path.join("output", args.project_name, "successful_refactor_codes.json")
        self.data_path = _input
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.project_path = os.path.join(args.project_dir)
        self.project_root = Path(args.project_dir)
        self.project_name = args.project_name
        self.project_repo = self.project_root / self.project_name
        self.cases = data['refactor_codes']
        self.settings = data['settings']
        self.src_path = self.settings['src_path']
        self.output_dir = Path(args.output_dir) / self.project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.limit = args.limit
        self.verbose = args.verbose
        self.use_code_agent = args.use_code_agent
        self.code_agent_command = args.code_agent_command
        self.llm_client: Optional[LLMClient] = None

        if not self.use_code_agent:
            self.llm_client = LLMFactory.create_client()
            
        self.results: List[CaseResult] = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _extract_content(self, caller_predictions: List[Segment], case):
        caller_content = []
        replaced_contents = defaultdict(list)
        original_contents = {}
        for pred in caller_predictions:
            pred_module = pred.meta['position']['module_path']
            for before_caller, file_content in zip(case['before_refactor_code'], case['caller_file_content']):
                if before_caller['module_path'] == pred_module:
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

    def _run_git_command(self, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=self.project_repo,
            text=True,
            capture_output=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Git command {' '.join(args)} failed with code {result.returncode}: {result.stderr.strip()}"
            )
        return result

    def _module_relative_path(self, module_path: str) -> Path:
        cleaned = (module_path or "").strip(".")
        src_tail = Path(self.src_path).parts[-1] if Path(self.src_path).parts else self.src_path
        if cleaned.startswith(f"{src_tail}."):
            cleaned = cleaned[len(src_tail) + 1 :]
        if not cleaned:
            rel = Path("__init__.py")
        else:
            rel = Path(cleaned.replace(".", os.sep) + ".py")
        return Path(self.src_path) / rel

    def _write_ground_truth_files(self, case: Dict[str, Any]) -> List[str]:
        written: List[str] = []
        for file_entry in case.get("caller_file_content", []):
            module_path = file_entry.get("module_path")
            if not module_path:
                continue
            rel_path = self._module_relative_path(module_path)
            abs_path = self.project_repo / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(file_entry.get("code", ""), encoding="utf-8")
            written.append(str(rel_path).replace("\\", "/"))
        return written

    def _has_staged_changes(self) -> bool:
        result = self._run_git_command(["diff", "--cached", "--quiet"], check=False)
        if result.returncode in (0, 1):
            return result.returncode == 1
        raise RuntimeError(f"Unexpected git diff --cached return code {result.returncode}")

    def _invoke_code_agent(self, prompt: str) -> None:
        command = shlex.split(self.code_agent_command)
        import shutil
        agent_cmd = shutil.which(command[0]).replace("/", os.sep)
        command[0] = agent_cmd
        process = subprocess.run(
            command,
            cwd=self.project_repo,
            input=prompt,
            text=True,
            capture_output=True,
        )
        if process.stdout:
            self._log(process.stdout.strip())
        if process.stderr:
            self._log(process.stderr.strip())
        if process.returncode != 0:
            raise RuntimeError(
                f"Code agent command {' '.join(command)} failed with code {process.returncode}"
            )

    def _cache_code_agent_diff(
        self,
        case_id: str,
        prompt_hash: str,
        diff_text: str,
        diff_files: List[str],
    ) -> None:
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        payload = {
            "case_id": case_id,
            "prompt_hash": prompt_hash,
            "project_name": self.project_name,
            "diff_files": diff_files,
            "diff": diff_text,
        }
        cache_path = cache_dir / f"code_agent_{self.project_name}_{case_id}.json"
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _modules_from_paths(self, diff_files: List[str]) -> Set[str]:
        modules: Set[str] = set()
        src_root = Path(self.src_path)
        base_module = src_root.parts[-1] if src_root.parts else src_root.name
        for diff_file in diff_files:
            rel_path = Path(diff_file.strip())
            try:
                relative = rel_path.relative_to(src_root)
            except ValueError:
                continue
            module_name = ".".join([base_module] + list(relative.with_suffix("").parts))
            modules.add(module_name)
        return modules

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
    ) -> PredictionArtifacts:
        changed_modules = self._modules_from_paths(diff_files)
        target_lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in case.get("before_refactor_code", []):
            module_path = entry.get("module_path")
            if module_path:
                target_lookup[module_path].append(entry)
        functions: List[FunctionSnippet] = []
        caller_segments: List[Segment] = []
        callee_segments: List[Segment] = []
        for file_entry in case.get("caller_file_content", []):
            module_path = file_entry.get("module_path")
            if not module_path:
                continue
            if changed_modules and module_path not in changed_modules:
                continue
            rel_path = self._module_relative_path(module_path)
            abs_path = self.project_repo / rel_path
            if not abs_path.exists():
                continue
            after_code = abs_path.read_text(encoding="utf-8")
            before_code = file_entry.get("code", "")
            after_functions = _extract_functions_from_block(after_code)
            before_functions = _extract_functions_from_block(before_code)
            functions.extend(after_functions)
            before_simple = {fn.name.split(".")[-1] for fn in before_functions}
            target_entries = target_lookup.get(module_path, [])
            target_names = {entry.get("method_name") for entry in target_entries}
            for entry in target_entries:
                match = self._match_function(entry, after_functions)
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
            for fn in after_functions:
                simple_name = fn.name.split(".")[-1]
                if simple_name in target_names:
                    continue
                if simple_name in before_simple:
                    continue
                callee_segments.append(
                    Segment(
                        text=fn.source,
                        meta={"type": "callee", "module_path": module_path},
                        name=simple_name,
                    )
                )
        return PredictionArtifacts(
            functions=functions,
            caller_segments=caller_segments,
            callee_segments=callee_segments,
            test_passed=None,
            raw_response=diff_text,
            parsed_payload={
                "backend": "code_agent",
                "changed_modules": sorted(changed_modules),
            },
        )

    def _run_code_agent_workflow(
        self,
        case_id: str,
        case: Dict[str, Any],
        prompt: str,
        prompt_hash: str,
    ) -> Tuple[PredictionArtifacts, bool]:
        original_head = self._run_git_command(["rev-parse", "HEAD"]).stdout.strip()
        diff_text = ""
        diff_files: List[str] = []
        success = False
        try:
            self._run_git_command(["reset", "--hard", original_head])
            written_files = self._write_ground_truth_files(case)
            if written_files:
                self._run_git_command(["add", *written_files])
                if self._has_staged_changes():
                    self._run_git_command(["commit", "-m", f"[baseline] {case_id}"])
            self._invoke_code_agent(prompt)
            diff_text = self._run_git_command(["diff"]).stdout
            diff_output = self._run_git_command(["diff", "--name-only"]).stdout
            diff_files = [line.strip() for line in diff_output.splitlines() if line.strip()]
            self._cache_code_agent_diff(case_id, prompt_hash, diff_text, diff_files)
            prediction = self._build_prediction_from_repo(case, diff_files, diff_text)
            success = replace_and_test_caller(
                self.project_name,
                self.src_path,
                case["testsuites"],
                None,
                self.project_path,
            )
        finally:
            self._run_git_command(["reset", "--hard", original_head])
        return prediction, success

    def run(self) -> Dict[str, Any]:
        total_cases = len(self.cases)
        self._log(f"Loaded {total_cases} cases from {self.data_path}")
        for index, case in enumerate(self.cases):
            if self.limit is not None and index >= self.limit:
                break
            case_id = case_identifier(case, index)
            self._log(f"Processing {case_id}")
            result = self._process_case(case_id, case)
            if result is None:
                continue
            self.results.append(result)
        per_case_path = self.output_dir / "per_case_results.json"
        serialized = [dataclasses.asdict(result) for result in self.results]
        per_case_path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = self._summarize()
        summary_path = self.output_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _process_case(self, case_id: str, case: Dict[str, Any]) -> CaseResult:
        if case['type'] == 'LongMethod':
            return
        prompt = build_prompt(case)
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()

        ground_truth = parse_ground_truth(case)
        if self.use_code_agent:
            prediction, success = self._run_code_agent_workflow(case_id, case, prompt, prompt_hash)
        else:
            payload = self._predict(prompt, prompt_hash, case_id, case)
            prediction = parse_model_prediction(payload["response_text"], case)
            caller_content = self._extract_content(prediction.caller_segments, case)
            success = replace_and_test_caller(
                self.project_name,
                self.src_path,
                case['testsuites'],
                caller_content,
                self.project_path
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
            print(f"No caller predictions found for case {case_id}.")
            return CaseResult(
                case_id=case_id,
                prompt_hash=prompt_hash,
                callee_precision=callee_precision,
                callee_recall=callee_recall,
                callee_f1=callee_f1,
                match_scores=0.0,
                test_passed=False
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
        return CaseResult(
            case_id=case_id,
            prompt_hash=prompt_hash,
            callee_precision=callee_precision,
            callee_recall=callee_recall,
            callee_f1=callee_f1,
            callee_match_score=match_scores,
            test_passed=success,
            details=details,
        )

    def _predict(
        self,
        prompt: str,
        prompt_hash: str,
        case_id: str,
        case: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            raise RuntimeError(f"Cache miss for {case_id} and no LLM client available.")
        response_text = self.llm_client.chat(
            prompt=prompt,
            model=self.args.model,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
        )
        payload = {
            "prompt_hash": prompt_hash,
            "model": self.args.model,
            "case_id": case_id,
            "prompt": prompt,
            "response_text": response_text,
            "metadata": {
                "project_name": case.get("project_name"),
                "module_path": case.get("module_path"),
            },
        }
        return payload

    def _summarize(self) -> Dict[str, Any]:
        summary = {
            "cases_evaluated": len(self.results),
            "output_dir": str(self.output_dir),
        }
        summary["averages"] = {
            "callee_match_score": mean_or_none(result.callee_match_score for result in self.results),
            "callee_precision": mean_or_none(result.callee_precision for result in self.results),
            "callee_recall": mean_or_none(result.callee_recall for result in self.results),
            "callee_f1": mean_or_none(result.callee_f1 for result in self.results),
        }
        test_values = [result.test_passed for result in self.results if result.test_passed is not None]
        if test_values:
            summary["test_pass_rate"] = sum(1 for value in test_values if value) / len(test_values)
        else:
            summary["test_pass_rate"] = None
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM refactor ability against reference data.")
    parser.add_argument("--output-dir", default="run/refactor_eval", help="Directory for cached outputs and reports.")
    parser.add_argument("--model", default=None, help="Model name used for generation.")
    parser.add_argument("--project-dir", default="../project", help="Project directory for resolving relative paths in test commands.")
    parser.add_argument("--project-name", default="click", help="Project name")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for the chat model.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Maximum tokens requested from the chat model.")
    parser.add_argument("--use-code-agent", action="store_true", help="Use code agent (Claude Code) instead of text-only LLM predictions.")
    parser.add_argument("--code-agent-command", default="claude -p --permission-mode acceptEdits", help="Command used to invoke the code agent. The prompt is sent via stdin.")
    parser.add_argument("--limit", type=int, help="Process at most this many cases.")
    parser.add_argument("--similarity-threshold", type=float, default=0.4, help="Similarity threshold used for F1 matching.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress information.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.use_code_agent = True
    evaluator = RefactorEvaluator(args)
    summary = evaluator.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
