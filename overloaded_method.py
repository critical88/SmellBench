import ast
import os
import random
import textwrap

from collections import defaultdict

from base_method import BaseCollector


class OverloadedMethodCollector(BaseCollector):
    def __init__(self, project_path, project_name, src_path) -> None:
        super().__init__(project_path, project_name, src_path)
        self.max_callers_per_variant = 5

    def _normalize_method_source(self, source: str) -> str:
        source = textwrap.dedent(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return "\n".join(line.strip() for line in source.splitlines() if line.strip())
        if not tree.body:
            return ""
        func_node = tree.body[0]
        if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if func_node.body and isinstance(func_node.body[0], ast.Expr):
                value = getattr(func_node.body[0], "value", None)
                if isinstance(value, ast.Str) or (
                    isinstance(value, ast.Constant) and isinstance(value.value, str)
                ):
                    func_node.body = func_node.body[1:]
            ast.fix_missing_locations(func_node)
            return ast.unparse(func_node)
        return "\n".join(ast.unparse(stmt) for stmt in tree.body)

    def _find_overloaded_roots(self, all_definitions, all_class_parents):
        overload_map = defaultdict(set)
        normalized_cache = {}

        def get_normalized(key):
            if key in normalized_cache:
                return normalized_cache[key]
            definition = all_definitions.get(key)
            if not definition or not definition.get("source"):
                normalized_cache[key] = ""
                return ""
            normalized_cache[key] = self._normalize_method_source(definition["source"])
            return normalized_cache[key]

        for method_key, definition in all_definitions.items():
            module_path, class_name, method_name = method_key
            if class_name is None or not definition.get("source"):
                continue
            parents = all_class_parents.get((module_path, class_name), [])
            if not parents:
                continue
            child_norm = get_normalized(method_key)
            if not child_norm:
                continue

            for parent_module, parent_class in parents:
                parent_key = (parent_module, parent_class, method_name)
                parent_def = all_definitions.get(parent_key)
                if not parent_def or not parent_def.get("source"):
                    continue
                parent_norm = get_normalized(parent_key)
                if not parent_norm or parent_norm == child_norm:
                    continue
                overload_map[parent_key].add(method_key)
        return overload_map

    def collect(self, class_methods, all_calls, all_definitions, all_class_parents, family_classes):
        overloaded_methods = []
        overload_roots = self._find_overloaded_roots(all_definitions, all_class_parents)

        for parent_key, child_keys in overload_roots.items():
            variant_entries = []
            for method_key in [parent_key] + sorted(child_keys):
                callers = class_methods.get(method_key)
                if callers:
                    variant_entries.append((method_key, callers))
            # Need at least parent and one child with callers
            if len(variant_entries) < 2:
                continue

            caller_replacement_dict = defaultdict(dict)
            after_refactor_code = []
            after_refactor_lookup = {}
            before_refactor_code = []
            caller_file_contents = []
            testsuites = set()
            valid_calling_times = 0

            for method_key, callers in variant_entries:
                definition = all_definitions.get(method_key)
                if not definition or not definition.get("source"):
                    continue
                callee_file = self.get_file_from_module(method_key[0])
                callee = {
                    "source": definition["source"],
                    "decorators": definition["decorators"],
                    "file": callee_file,
                }
                callee["position"] = {
                    "module_path": method_key[0],
                    "class_name": method_key[1],
                    "method_name": method_key[2],
                }

                caller_candidates = callers
                if len(caller_candidates) > self.max_callers_per_variant:
                    caller_candidates = random.sample(caller_candidates, self.max_callers_per_variant)

                for caller_composite in caller_candidates:
                    caller_method = caller_composite["position"]
                    caller_module = caller_method[0]
                    caller_file = self.get_file_from_module(caller_module)
                    if not os.path.exists(caller_file):
                        continue
                    caller_method_definition = all_definitions.get(caller_method)
                    if caller_method_definition is None:
                        continue

                    for caller_location in caller_composite["call_locations"]:
                        caller = caller_location.copy()
                        caller["source"] = caller_method_definition["source"]
                        caller["position"] = {
                            "module_path": caller_method[0],
                            "class_name": caller_method[1],
                            "method_name": caller_method[2],
                        }
                        caller["file"] = caller_file
                        caller_replacement = self.generate_replacement_caller_from_callee(
                            caller, callee, all_class_parents
                        )
                        if caller_replacement is None:
                            continue

                        if isinstance(caller_replacement["replacement"], str):
                            caller_replacement["replacement"] = caller_replacement["replacement"].splitlines()
                        caller_replacement["lines"] = len(definition["source"].splitlines())
                        rel_start = caller_replacement["start"] - caller_method_definition["start_line"] + 1
                        rel_end = caller_replacement["end"] - caller_method_definition["start_line"] + 1
                        caller_replacement["rel_start"] = rel_start
                        caller_replacement["rel_end"] = rel_end

                        caller_replacements = caller_replacement_dict[caller_module].setdefault(caller_method, [])
                        caller_replacements.append(caller_replacement)
                        valid_calling_times += 1
                        testsuites.update(self._find_related_testsuite(caller_method))

                        after_key = (caller_module, caller_method)
                        if after_key not in after_refactor_lookup:
                            after_refactor = {
                                "type": "caller",
                                "code": caller_method_definition["source"],
                                "position": {
                                    "module_path": caller_method[0],
                                    "class_name": caller_method[1],
                                    "method_name": caller_method[2],
                                },
                                "callees": [],
                            }
                            after_refactor_lookup[after_key] = after_refactor
                            after_refactor_code.append(after_refactor)
                        callee_entry = {
                            "type": "callee",
                            "decorators": definition["decorators"],
                            "start": caller_replacement["start"],
                            "end": caller_replacement["end"],
                            "code": definition["source"],
                            "position": callee["position"],
                        }
                        after_refactor_lookup[after_key]["callees"].append(callee_entry)

            if not caller_replacement_dict:
                continue

            for caller_module, replacements in caller_replacement_dict.items():
                before_refactors, caller_lines = self.do_replacement(replacements, caller_module, all_definitions)
                if before_refactors:
                    before_refactor_code.extend(before_refactors)
                    caller_file_contents.append(
                        {
                            "code": "\n".join(caller_lines),
                            "module_path": caller_module,
                            "file_suffix": f"overload_{parent_key[2]}",
                        }
                    )

            if not before_refactor_code:
                continue

            variant_classes = [
                f"{method_key[0]}.{method_key[1]}" if method_key[1] else method_key[0]
                for method_key, _ in variant_entries
            ]
            overloaded_methods.append(
                {
                    "type": "OverloadedMethod",
                    "meta": {
                        "method_name": parent_key[2],
                        "base_class": f"{parent_key[0]}.{parent_key[1]}",
                        "variant_classes": variant_classes,
                        "calling_times": valid_calling_times,
                    },
                    "testsuites": list(testsuites),
                    "after_refactor_code": after_refactor_code,
                    "before_refactor_code": before_refactor_code,
                    "caller_file_content": caller_file_contents,
                }
            )

        return overloaded_methods
