#!/usr/bin/env python
"""
Build a method-to-test tree for click by running coverage with per-test contexts.

This script discovers and runs the pytest suite that lives in ``../project/click`` (relative
to this file), records which implementation methods are exercised by each test, and prints
an ASCII tree whose leaves are the test cases. Give it a method key and it will list all
tests connected to that method.
"""
from __future__ import annotations

import argparse
import ast

import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set
import subprocess
from types import CodeType, FrameType
from utils import pushd, prepare_to_run, get_spec
import shlex

try:
    import coverage  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "The coverage package is required. Install it with `pip install coverage`."
    ) from exc

try:
    import pytest  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("pytest is required to run the click tests.") from exc


@dataclass(frozen=True)
class FunctionInfo:
    """Metadata for a single function or method."""

    module: str
    qualname: str
    filepath: Path
    start: int
    end: int
    variable_types: Dict[str, Set[str]]

    @property
    def key(self) -> str:
        """Return the canonical lookup key for this function."""
        return f"{self.module}:{self.qualname}"


@dataclass(frozen=True)
class ClassInfo:
    """Metadata describing a discovered class and its parent names."""

    module: str
    qualname: str
    bases: List[str]

    @property
    def key(self) -> str:
        return f"{self.module}:{self.qualname}"


@dataclass
class FileFunctionIndex:
    """Line-to-function lookup for a source file."""

    functions: List[FunctionInfo]
    lines_to_functions: Dict[int, List[FunctionInfo]]
    classes: List[ClassInfo]

class ModuleMetadataCollector(ast.NodeVisitor):
    """Collect import aliases and class names for a module."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.imports: Dict[str, str] = {}
        self.class_names: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:  # pragma: no cover - AST plumbing
        for alias in node.names:
            alias_name = alias.asname or alias.name
            self.imports[alias_name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pragma: no cover - AST plumbing
        if node.level > 0:  
            parts = self.module.split('.')
            if node.level > len(parts):
                return  
            base = '.'.join(parts[:-node.level])
            if node.module:  # from .foo import bar
                base = f"{base}.{node.module}" if base else node.module
        else:  
            base = node.module or ''
        for alias in node.names:
            if alias.name == '*':
                continue  
            module_name = f"{base}.{alias.name}" if base else alias.name
            asname = alias.asname or alias.name
            self.imports[asname] = module_name

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - AST plumbing
        self.class_names.add(node.name)
        self.generic_visit(node)


class VariableTypeExtractor(ast.NodeVisitor):
    """Extract variable type information for a single function body."""

    def __init__(
        self,
        module: str,
        project_name: str,
        imports: Mapping[str, str],
        local_classes: Set[str],
        func_node: ast.AST,
    ) -> None:
        self.module = module
        self.project_name = project_name
        self.imports = imports
        self.local_classes = local_classes
        self.func_node = func_node
        self.variable_types: Dict[str, Set[str]] = defaultdict(set)
        self._known_types: Dict[str, Set[str]] = defaultdict(set)

    def collect(self) -> Dict[str, Set[str]]:
        """Return a mapping of variable name -> package-qualified class types."""
        if not isinstance(self.func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {}
        self._capture_parameters(self.func_node)
        for stmt in self.func_node.body:
            self.visit(stmt)
        return {name: set(types) for name, types in self.variable_types.items()}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - explicit control
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover - explicit control
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - explicit control
        return

    def visit_Assign(self, node: ast.Assign) -> None:
        types = self._infer_value_types(node.value)
        if types:
            for target in node.targets:
                self._assign_target(target, types)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        types = self._resolve_types(node.annotation)
        if not types:
            types = self._infer_value_types(node.value)
        if types:
            self._assign_target(node.target, types)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # Reuse existing type information on augmented assignments.
        types = self._infer_value_types(node.value)
        if types:
            self._assign_target(node.target, types)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            types = self._infer_value_types(item.context_expr)
            if types and item.optional_vars:
                self._assign_target(item.optional_vars, types)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            types = self._infer_value_types(item.context_expr)
            if types and item.optional_vars:
                self._assign_target(item.optional_vars, types)
        self.generic_visit(node)


class VariableTypeExtractor(ast.NodeVisitor):
    """Extract variable type information for a single function body."""

    def __init__(
        self,
        module: str,
        project_name: str,
        imports: Mapping[str, str],
        local_classes: Set[str],
        func_node: ast.AST,
    ) -> None:
        self.module = module
        self.project_name = project_name
        self.imports = imports
        self.local_classes = local_classes
        self.func_node = func_node
        self.variable_types: Dict[str, Set[str]] = defaultdict(set)
        self._known_types: Dict[str, Set[str]] = defaultdict(set)

    def collect(self) -> Dict[str, Set[str]]:
        """Return a mapping of variable name -> package-qualified class types."""
        if not isinstance(self.func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {}
        self._capture_parameters(self.func_node)
        for stmt in self.func_node.body:
            self.visit(stmt)
        return {name: set(types) for name, types in self.variable_types.items()}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - explicit control
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover - explicit control
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - explicit control
        return

    def visit_Assign(self, node: ast.Assign) -> None:
        types = self._infer_value_types(node.value)
        if types:
            for target in node.targets:
                self._assign_target(target, types)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        types = self._resolve_types(node.annotation)
        if not types:
            types = self._infer_value_types(node.value)
        if types:
            self._assign_target(node.target, types)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # Reuse existing type information on augmented assignments.
        types = self._infer_value_types(node.value)
        if types:
            self._assign_target(node.target, types)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            types = self._infer_value_types(item.context_expr)
            if types and item.optional_vars:
                self._assign_target(item.optional_vars, types)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            types = self._infer_value_types(item.context_expr)
            if types and item.optional_vars:
                self._assign_target(item.optional_vars, types)
        self.generic_visit(node)

    def _capture_parameters(self, node: ast.AST) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        all_args = list(getattr(node.args, "posonlyargs", [])) + node.args.args
        for arg in all_args:
            self._record_arg_type(arg)
        if node.args.vararg:
            self._record_arg_type(node.args.vararg)
        for arg in node.args.kwonlyargs:
            self._record_arg_type(arg)
        if node.args.kwarg:
            self._record_arg_type(node.args.kwarg)

    def _record_arg_type(self, arg: ast.arg) -> None:
        if not isinstance(arg, ast.arg):
            return
        for type_name in self._resolve_types(arg.annotation):
            self._add_type(arg.arg, type_name)

    def _assign_target(self, target: ast.AST, types: Set[str]) -> None:
        if isinstance(target, ast.Name):
            for type_name in types:
                self._add_type(target.id, type_name)
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                attr_name = f"self.{target.attr}"
                for type_name in types:
                    self._add_type(attr_name, type_name)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                self._assign_target(elt, types)

    def _infer_value_types(self, value: Optional[ast.AST]) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, ast.Call):
            return self._resolve_types(value.func)
        if isinstance(value, ast.Name):
            return set(self._known_types.get(value.id, set()))
        if isinstance(value, ast.Attribute):
            if isinstance(value.value, ast.Name) and value.value.id == "self":
                key = f"self.{value.attr}"
                return set(self._known_types.get(key, set()))
        return set()

    def _add_type(self, var_name: str, type_name: str) -> None:
        if not type_name or not type_name.startswith(f"{self.project_name}."):
            return
        self.variable_types[var_name].add(type_name)
        self._known_types[var_name].add(type_name)

    def _resolve_types(self, node: Optional[ast.AST]) -> Set[str]:
        if node is None:
            return set()
        if isinstance(node, ast.Subscript):
            return self._resolve_types(node.value)
        if hasattr(ast, "Index") and isinstance(node, ast.Index):  # pragma: no cover - py<3.9
            return self._resolve_types(node.value)
        if isinstance(node, ast.Tuple):
            resolved: Set[str] = set()
            for elt in node.elts:
                resolved.update(self._resolve_types(elt))
            return resolved
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            resolved: Set[str] = set()
            resolved.update(self._resolve_types(node.left))
            resolved.update(self._resolve_types(node.right))
            return resolved
        if isinstance(node, (ast.List, ast.Set)):
            resolved: Set[str] = set()
            for elt in node.elts:
                resolved.update(self._resolve_types(elt))
            return resolved
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self._resolve_types_from_string(node.value)
        if hasattr(ast, "Str") and isinstance(node, ast.Str):  # pragma: no cover - py<3.8
            return self._resolve_types_from_string(node.s)
        chain = self._get_name_chain(node)
        if not chain:
            return set()
        type_name = self._resolve_name_chain(chain)
        return {type_name} if type_name else set()

    def _resolve_types_from_string(self, value: Optional[str]) -> Set[str]:
        if not value:
            return set()
        chain = value.split(".")
        type_name = self._resolve_name_chain(chain)
        return {type_name} if type_name else set()

    def _get_name_chain(self, node: ast.AST) -> Optional[List[str]]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            base = self._get_name_chain(node.value)
            if base is None:
                return None
            return base + [node.attr]
        
    def _resolve_name_chain(self, chain: List[str]) -> Optional[str]:
        if not chain:
            return None
        base = chain[0]
        rest = chain[1:]
        resolved = None
        if base in self.imports:
            resolved = self.imports[base]
        elif base == self.project_name:
            resolved = base
        elif base in self.local_classes:
            resolved = f"{self.module}.{base}"
        elif ".".join(chain).startswith(f"{self.project_name}."):
            resolved = ".".join(chain)
        if resolved is None:
            return None
        if rest:
            resolved = f"{resolved}.{'.'.join(rest)}"
        return resolved if resolved.startswith(f"{self.project_name}.") else None

class FunctionCollector(ast.NodeVisitor):
    """Collect nested function/method definitions and class hierarchies."""

    def __init__(self, 
                 module: str, 
                 filepath: Path, 
                 project_name: str,
                 imports: Mapping[str, str],
                 local_classes: Set[str]
                 ) -> None:
        self.module = module
        self.filepath = filepath.resolve()
        self.project_name = project_name
        self.imports = imports
        self.local_classes = local_classes
        self._scope: List[str] = []
        self.functions: List[FunctionInfo] = []
        self.class_infos: List[ClassInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join(self._scope + [node.name])
        bases = [name for name in (self._resolve_class_from_name(base) for base in node.bases) if name]
        
        self.class_infos.append(ClassInfo(self.module, qualname, bases))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _resolve_class_from_name(self, node):
        name_chain = self._get_name_chain(node)
        if not name_chain:
            return None
        if len(name_chain) == 1 and name_chain[0] in self.local_classes:
            return self.module, name_chain[0]
        for i in range(len(name_chain), 0, -1):
            prefix = ".".join(name_chain[:i])
            if prefix in self.imports:
                base = self.imports[prefix]
                rest = name_chain[i:]
                if rest:
                    full_path = f"{base}.{'.'.join(rest)}"
                else:
                    full_path = base
                module_path, class_name = self._split_module_and_class(full_path)
                if module_path and class_name and module_path.startswith(self.project_name):
                    return module_path, class_name
                return None
        return None
    def _split_module_and_class(self, full_path):
        if not full_path or "." not in full_path:
            return None, None
        parts = full_path.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        if not module_path or not class_name:
            return None, None
        return module_path, class_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)
    
    def _record_function(self, node: ast.AST) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        qualname = ".".join(self._scope + [node.name])
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            end_lineno = self._infer_end_lineno(node)
        
        var_types = VariableTypeExtractor(
            self.module,
            self.project_name,
            self.imports,
            self.local_classes,
            node,
        ).collect()
        info = FunctionInfo(
            module=self.module,
            qualname=qualname,
            filepath=self.filepath,
            start=node.lineno,
            end=end_lineno,
            variable_types=var_types
        )
        self.functions.append(info)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()
    
    def _capture_parameters(self, node: ast.AST) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        all_args = list(getattr(node.args, "posonlyargs", [])) + node.args.args
        for arg in all_args:
            self._record_arg_type(arg)
        if node.args.vararg:
            self._record_arg_type(node.args.vararg)
        for arg in node.args.kwonlyargs:
            self._record_arg_type(arg)
        if node.args.kwarg:
            self._record_arg_type(node.args.kwarg)

    def _record_arg_type(self, arg: ast.arg) -> None:
        if not isinstance(arg, ast.arg):
            return
        for type_name in self._resolve_types(arg.annotation):
            self._add_type(arg.arg, type_name)

    def _assign_target(self, target: ast.AST, types: Set[str]) -> None:
        if isinstance(target, ast.Name):
            for type_name in types:
                self._add_type(target.id, type_name)
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                attr_name = f"self.{target.attr}"
                for type_name in types:
                    self._add_type(attr_name, type_name)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                self._assign_target(elt, types)

    def _infer_value_types(self, value: Optional[ast.AST]) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, ast.Call):
            return self._resolve_types(value.func)
        if isinstance(value, ast.Name):
            return set(self._known_types.get(value.id, set()))
        if isinstance(value, ast.Attribute):
            if isinstance(value.value, ast.Name) and value.value.id == "self":
                key = f"self.{value.attr}"
                return set(self._known_types.get(key, set()))
        return set()

    def _add_type(self, var_name: str, type_name: str) -> None:
        if not type_name or not type_name.startswith(f"{self.project_name}."):
            return
        self.variable_types[var_name].add(type_name)
        self._known_types[var_name].add(type_name)

    def _resolve_types(self, node: Optional[ast.AST]) -> Set[str]:
        if node is None:
            return set()
        if isinstance(node, ast.Subscript):
            types = self._resolve_types(node.slice)
            if types:
                return types
            return self._resolve_types(node.value)
        if hasattr(ast, "Index") and isinstance(node, ast.Index):  # pragma: no cover - py<3.9
            return self._resolve_types(node.value)
        if isinstance(node, ast.Tuple):
            resolved: Set[str] = set()
            for elt in node.elts:
                resolved.update(self._resolve_types(elt))
            return resolved
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            resolved: Set[str] = set()
            resolved.update(self._resolve_types(node.left))
            resolved.update(self._resolve_types(node.right))
            return resolved
        if isinstance(node, (ast.List, ast.Set)):
            resolved: Set[str] = set()
            for elt in node.elts:
                resolved.update(self._resolve_types(elt))
            return resolved
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self._resolve_types_from_string(node.value)
        if hasattr(ast, "Str") and isinstance(node, ast.Str):  # pragma: no cover - py<3.8
            return self._resolve_types_from_string(node.s)
        chain = self._get_name_chain(node)
        if not chain:
            return set()
        type_name = self._resolve_name_chain(chain)
        return {type_name} if type_name else set()

    def _resolve_types_from_string(self, value: Optional[str]) -> Set[str]:
        if not value:
            return set()
        chain = value.split(".")
        type_name = self._resolve_name_chain(chain)
        return {type_name} if type_name else set()

    def _get_name_chain(self, node: ast.AST) -> Optional[List[str]]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            base = self._get_name_chain(node.value)
            if base is None:
                return None
            return base + [node.attr]

    @staticmethod
    def _infer_end_lineno(node: ast.AST) -> int:
        max_lineno = getattr(node, "lineno", 0)
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                max_lineno = max(max_lineno, getattr(child, "lineno"))
        return max_lineno

    def _resolve_base_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                parts.reverse()
                return ".".join(parts)
            return None
        if isinstance(node, ast.Subscript):
            return self._resolve_base_name(node.value)
        try:
            return ast.unparse(node)
        except Exception:
            return None


class CoverageGraph:
    """Convert coverage data with per-test contexts into a tree."""

    def __init__(
        self,
        src_root: Path,
        file_index: Mapping[Path, FileFunctionIndex],
        lookup: Mapping[str, FunctionInfo],
        class_lookup: Optional[Mapping[str, ClassInfo]] = None,
        direct_calls: Optional[Mapping[str, Set[str]]] = None,
    ) -> None:
        """Initialize the graph with source metadata and empty adjacency lists."""
        self.src_root = src_root.parent
        self.file_index = file_index
        self.lookup = lookup
        self.class_lookup = class_lookup or {}
        self.direct_calls = direct_calls
        self.function_to_tests: Dict[str, Set[str]] = defaultdict(set)
        self.test_to_functions: Dict[str, Set[str]] = defaultdict(set)

    def merge(self, data: coverage.CoverageData) -> None:
        """Populate adjacency lists by replaying coverage line→context data."""
        if not hasattr(data, "contexts_by_lineno"):
            raise SystemExit(
                "Coverage was run without per-test contexts. Upgrade coverage>=5."
            )
        for filename in data.measured_files():
            path = Path(filename).resolve()
            try:
                path.relative_to(self.src_root)
            except ValueError:
                continue
            index = self.file_index.get(path)
            if not index:
                continue
            contexts_map = data.contexts_by_lineno(filename)
            if not contexts_map:
                continue
            for line_no, contexts in contexts_map.items():
                funcs = index.lines_to_functions.get(line_no)
                if not funcs:
                    continue
                for context in contexts:
                    if not context or not _looks_like_test_context(context):
                        continue
                    normalized = _normalize_test_context(context)
                    if not normalized:
                        continue
                    allowed: Optional[Set[str]] = None
                    if self.direct_calls is not None:
                        allowed = self.direct_calls.get(normalized)
                        if not allowed:
                            continue
                    for func in funcs:
                        if allowed is not None and func.key not in allowed:
                            continue
                        self.function_to_tests[func.key].add(normalized)
                        # self.test_to_functions[normalized].add(func.key)

    def build_tree(self) -> Dict[str, MutableMapping[str, object]]:
        """Return a nested dict module→class→method with attached test leaves."""
        tree: Dict[str, MutableMapping[str, object]] = {}
        for func_key, tests in self.function_to_tests.items():
            module, qual = func_key.split(":", 1)
            node = tree.setdefault(module, {})
            for part in qual.split("."):
                node = node.setdefault(part, {})  # type: ignore[assignment]
            node.setdefault("__tests__", set())  # type: ignore[index]
            node["__tests__"].update(tests)  # type: ignore[index]

        def finalize(subtree: MutableMapping[str, object]) -> None:
            """Convert set leaves to sorted lists for reproducible output."""
            tests = subtree.get("__tests__")
            if isinstance(tests, set):
                subtree["__tests__"] = sorted(tests)
            for key, child in list(subtree.items()):
                if key == "__tests__":
                    continue
                if isinstance(child, MutableMapping):
                    finalize(child)

        for module_name, module_tree in tree.items():
            if isinstance(module_tree, MutableMapping):
                finalize(module_tree)
        return tree

    def print_tree(self, tree: Mapping[str, object]) -> None:
        """Pretty-print tree nodes using indentation."""
        if not tree:
            print("No methods were executed by the tests.")
            return
        for module in sorted(tree):
            print(module)
            node = tree[module]
            if isinstance(node, Mapping):
                self._print_subtree(node, indent=1)

    def _print_subtree(self, subtree: Mapping[str, object], indent: int) -> None:
        """Recursive helper for print_tree that handles child nodes and leaves."""
        pad = "  " * indent
        child_keys = sorted(key for key in subtree.keys() if key != "__tests__")
        for key in child_keys:
            print(f"{pad}{key}")
            child = subtree[key]
            if isinstance(child, Mapping):
                self._print_subtree(child, indent + 1)
        tests = subtree.get("__tests__")
        if isinstance(tests, list):
            for test in tests:
                print(f"{pad}- [test] {test}")

    def print_method_matches(self, pattern: str, fuzzy: bool = False) -> None:
        """Display tests associated with matching methods."""
        matches = self.find_method_matches(pattern, fuzzy=fuzzy)
        if not matches:
            scope = "substring" if fuzzy else "exact"
            print(f"No {scope} matches found for '{pattern}'.")
            return
        for method in sorted(matches):
            info = self.lookup.get(method)
            location = ""
            if info:
                rel = info.filepath.relative_to(self.src_root)
                location = f" ({rel}:{info.start})"
            print(f"{method}{location}")
            for test in sorted(self.function_to_tests.get(method, [])):
                print(f"  - {test}")

    def find_method_matches(self, pattern: str, fuzzy: bool = False) -> Set[str]:
        """Return set of method keys matching either exactly or by substring."""
        if not fuzzy:
            return {pattern} if pattern in self.function_to_tests else set()
        needle = pattern.lower()
        return {
            method
            for method in self.function_to_tests
            if needle in method.lower()
        }

    def export_json(self, path: Path, meta={}) -> None:
        """Serialize method/test adjacency info for downstream tooling."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": meta,
            "functions": {
                method: {
                    "file": str(info.filepath),
                    "relative_file": str(info.filepath.relative_to(self.src_root)),
                    "line_range": [info.start, info.end],
                    "tests": sorted(self.function_to_tests[method]),
                    "variable_types": {var: sorted(types) for var, types in info.variable_types.items()}
                }
                for method, info in self.lookup.items()
                if method in self.function_to_tests
            },
            # "tests": {
            #     test: sorted(functions)
            #     for test, functions in self.test_to_functions.items()
            # },
        }
        if self.class_lookup:
            payload["classes"] = {
                key: {
                    "module": info.module,
                    "qualname": info.qualname,
                    "bases": info.bases,
                }
                for key, info in self.class_lookup.items()
            }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote JSON graph to {path}")


def _looks_like_test_context(name: str) -> bool:
    """Heuristically detect pytest test nodeids (skip internal coverage contexts)."""
    return "::" in name and not name.startswith("pytest:")


def _normalize_test_context(name: str) -> str:
    """Strip parameter details (``[]``) from pytest nodeids to deduplicate variants."""
    if not name:
        return ""
    bracket = name.find("[")
    return name[:bracket] if bracket != -1 else name


class _DirectCallTracer:
    """Profiler that records functions called directly from a test function body."""

    def __init__(
        self,
        test_code: CodeType,
        nodeid: str,
        code_lookup: Mapping[tuple[str, int], str],
        call_store: MutableMapping[str, Set[str]],
    ) -> None:
        self.test_code = test_code
        self.nodeid = nodeid
        self.code_lookup = code_lookup
        self.call_store = call_store
        self.previous_profile: Optional[
            Callable[[FrameType, str, object], None]
        ] = None
        self._active = False

    def start(self) -> Callable[[], None]:
        if not self.nodeid:
            return lambda: None
        self.previous_profile = sys.getprofile()
        sys.setprofile(self._trace)
        self._active = True
        return self.stop

    def stop(self) -> None:
        if not self._active:
            return
        sys.setprofile(self.previous_profile)
        self._active = False

    def _trace(self, frame: FrameType, event: str, arg) -> None:
        if event != "call":
            return
        caller = frame.f_back
        if not caller or caller.f_code is not self.test_code:
            return
        key = self._resolve_key(frame)
        if key:
            bucket = self.call_store.get(self.nodeid)
            if bucket is None:
                bucket = set()
                self.call_store[self.nodeid] = bucket
            bucket.add(key)

    def _resolve_key(self, frame: FrameType) -> Optional[str]:
        filename = Path(frame.f_code.co_filename).resolve()
        line = frame.f_code.co_firstlineno
        return self.code_lookup.get((str(filename), line))


def build_function_index(
    src_root: Path, project_name
) -> tuple[
    Dict[Path, FileFunctionIndex],
    Dict[str, FunctionInfo],
    Dict[str, ClassInfo],
    Dict[tuple[str, int], str],
]:
    """Walk the source tree and build lookup tables for every function and class definition."""
    file_index: Dict[Path, FileFunctionIndex] = {}
    function_lookup: Dict[str, FunctionInfo] = {}
    class_lookup: Dict[str, ClassInfo] = {}
    function_code_lookup: Dict[tuple[str, int], str] = {}
    for py_file in src_root.rglob("*.py"):
        if py_file.name.startswith("."):
            continue
        module = ".".join(py_file.relative_to(src_root.parent).with_suffix("").parts)
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError) as exc:
            print(f"Skipping {py_file}: {exc}", file=sys.stderr)
            continue
        metadata_collector = ModuleMetadataCollector(module)
        metadata_collector.visit(tree)
        collector = FunctionCollector(
            module,
            py_file,
            project_name,
            metadata_collector.imports,
            metadata_collector.class_names,
        )
        collector.visit(tree)
        if not collector.functions and not collector.class_infos:
            continue
        line_map: Dict[int, List[FunctionInfo]] = defaultdict(list)
        for func in collector.functions:
            function_lookup[func.key] = func
            function_code_lookup[(str(func.filepath), func.start)] = func.key
            for line in range(func.start, func.end + 1):
                line_map[line].append(func)
        for class_info in collector.class_infos:
            class_lookup[class_info.key] = class_info
        file_index[collector.filepath] = FileFunctionIndex(
            functions=collector.functions,
            lines_to_functions=line_map,
            classes=collector.class_infos,
        )
    if not file_index:
        raise SystemExit(f"No Python files discovered under {src_root}.")
    return file_index, function_lookup, class_lookup, function_code_lookup


class CoverageContextPlugin:
    """Pytest plugin that tracks direct calls and switches coverage contexts."""

    def __init__(
        self,
        cov: coverage.Coverage,
        code_lookup: Mapping[tuple[str, int], str],
        direct_calls: Optional[MutableMapping[str, Set[str]]],
        require_direct_calls: bool,
    ) -> None:
        self.coverage = cov
        self.code_lookup = code_lookup
        self.direct_calls = direct_calls
        self.require_direct_calls = require_direct_calls

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):  # type: ignore[override]
        """Track direct calls from the test body only during the call phase."""
        normalized = _normalize_test_context(item.nodeid)
        stop_profiler = self._start_direct_call_profiler(item, normalized)
        self.coverage.switch_context(item.nodeid)
        try:
            yield
        finally:
            self.coverage.switch_context("pytest:idle")
            if stop_profiler:
                stop_profiler()

    def _start_direct_call_profiler(
        self, item, normalized: str
    ) -> Optional[Callable[[], None]]:
        if not self.require_direct_calls or self.direct_calls is None:
            return None
        test_callable = getattr(item, "obj", None)
        code = getattr(test_callable, "__code__", None)
        if not isinstance(code, CodeType):
            return None
        tracer = _DirectCallTracer(code, normalized, self.code_lookup, self.direct_calls)
        return tracer.start()





def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Configure CLI arguments for controlling coverage execution and queries."""
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest with coverage contexts and build a tree that links click "
            "methods to the tests that execute them."
        )
    )
    parser.add_argument(
        "--project-name",
        default="click",
        type=str
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default= "../project",
        help="Path to the click project. Defaults to ../project relative to this file.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="pytest_args",
        help="Extra argument to forward to pytest. Specify multiple times as needed.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue building the graph even if pytest exits with a non-zero status.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Fully qualified method key (module:qualname) to list connected tests for.",
    )
    parser.add_argument(
        "--method-contains",
        action="append",
        help="Substring to fuzzy-match against method keys when listing tests.",
    )
    parser.add_argument(
        "--skip-tree",
        action="store_true",
        help="Skip printing the ASCII tree (useful when only querying methods).",
    )
    parser.add_argument(
        "--commit-id",
        type=str,
        default="HEAD"
        )
    parser.add_argument(
        "--direct-call-only",
        action="store_true",
        default=False,
        help="Only map methods that are directly called from within each pytest test.",
    )
    return parser.parse_args(argv)


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point: orchestrate coverage run, graph building, and reporting."""
    project_root =  args.project_root / args.project_name

    spec = get_spec(args.project_name)

    current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    target_conda_env = None
    if "env_name" in spec:
        target_conda_env = spec['env_name']
    if target_conda_env is not None and current_conda_env != target_conda_env:
        print(f"environment different, redirected to {target_conda_env}")
        # subprocess.run(["pytest", "-q"])
        # subprocess.run(["conda", "run", "-n", target_conda_env, "--live-stream", "pytest"])
        subprocess.run(["conda", "run", "-n", target_conda_env, "--live-stream", "python", "testunit_coverage.py", "--project-name", args.project_name], text=True)
        return 
    if not prepare_to_run(spec):
        return False
    
    commit_id = args.commit_id
    if not project_root.exists():
        raise SystemExit(f"Project root {project_root} does not exist.")
    src_root = (project_root / args.src_path).resolve()
    
    output_dir = Path("output") / args.project_name
    os.makedirs(output_dir, exist_ok=True)
    output_json = output_dir / "function_testunit_mapping.json"

    sys.path.insert(0, str(src_root.parent))
    sys.path.insert(0, str(project_root))

    (
        file_index,
        lookup,
        class_lookup,
        function_code_lookup,
    ) = build_function_index(src_root, args.project_name)
    cov = coverage.Coverage(branch=True, source=[str(src_root)])
    cov.erase()
    direct_call_map: Optional[MutableMapping[str, Set[str]]] = None
    if args.direct_call_only:
        direct_call_map = defaultdict(set)
    plugin = CoverageContextPlugin(
        cov,
        function_code_lookup,
        direct_call_map,
        require_direct_calls=args.direct_call_only,
    )
        
    pytest_args = []
    if args.pytest_args:
        pytest_args.extend(shlex.split(args.pytest_args, posix=True))
    envs = {}
    if args.src_path != args.project_name:
        envs["PYTHONPATH"] = str(Path(args.src_path).parent)
    exit_code = 0
    with pushd(project_root):
        subprocess.run(['git', 'reset', '--hard', commit_id], cwd=".", check=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=".", check=True)
        cov.start()
        print(f"start pytesting")
        try:
            exit_code = pytest.main(pytest_args, plugins=[plugin])
        finally:
            cov.stop()
            cov.save()
    print(f"pytest done")
    if exit_code != 0:
        message = f"pytest exited with status {exit_code}."
        # if not args.allow_failures:
        #     raise SystemExit(message)
        print(f"Warning: {message} Proceeding with partial data.", file=sys.stderr)
    print(f"start data collecting")
    data = cov.get_data()
    graph = CoverageGraph(
        src_root,
        file_index,
        lookup,
        class_lookup,
        direct_call_map,
    )
    graph.merge(data)
    tree = graph.build_tree()
    print(f"data collect done")
    print(
        f"Mapped {len(graph.function_to_tests)} methods across "
        f"{len(graph.test_to_functions)} tests."
    )
    meta = {
        "src_path": args.src_path,
        "commit_id": args.commit_id,
        "test_cmd": args.pytest_args,
        "envs": envs,
        "direct_call_only": args.direct_call_only,
    }

    graph.export_json(output_json, meta)

    # queries_ran = False
    # if args.method:
    #     for method in args.method:
    #         graph.print_method_matches(method, fuzzy=False)
    #     queries_ran = True
    # if args.method_contains:
    #     for pattern in args.method_contains:
    #         graph.print_method_matches(pattern, fuzzy=True)
    #     queries_ran = True

    # if not args.skip_tree and not queries_ran:
    #     graph.print_tree(tree)

    return 0


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    repo_spec = get_spec(args.project_name)
    repo_name = repo_spec['name']
    args.project_name = repo_spec['name']
    args.src_path = repo_spec['src_path'] if 'src_path' in repo_spec else f'src/{repo_name}'
    args.commit_id = repo_spec['commit_id'] if 'commit_id' in repo_spec else 'HEAD'
    args.pytest_args = repo_spec['test_cmd'] if "test_cmd" in repo_spec else ""
    main(args)
    raise SystemExit()
