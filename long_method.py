import ast
import os

from base_method import BaseCollector
from tqdm import tqdm
import random
from typing import Dict, Tuple, List
from utils import hashcode

class LongMethodCollector(BaseCollector):
    MINIMAL_CALLEE_NUM = 1
    TOTAL_CALLEE_LENGTH_THRESHOLD = 20

    def __init__(self, project_path: str, project_name: str, src_path: str, commitid:str, all_definitions, family_classes, max_inline_depth:int=1) -> None:
        super().__init__(project_path, project_name, src_path, commitid, all_definitions, family_classes)
        self._expanded_method_cache = {}
        self.max_inline_depth = max(1, max_inline_depth)

    def name(self):
        return "Long"
    
    def _expand_method_source(self, root_caller_method, method_key, all_calls, all_class_parents, depth=0, max_depth=1):

        remaining_depth = max(max_depth - depth, 0)
        if remaining_depth == 0:
            return
        cache_key = (method_key, remaining_depth)
        if cache_key in self._expanded_method_cache:
            return self._expanded_method_cache[cache_key]
        
        caller_method_definition = self.all_definitions.get(method_key)
        if caller_method_definition is None:
            return 
        
        caller_len = self._normalized_function_length(caller_method_definition['source'])
        if caller_len < 3:
            return
        
        callee_methods = all_calls.get(method_key, {})
        caller_module = method_key[0]
        caller_file = self.get_file_from_module(caller_module)
        callees = []
        imports = []
        testunits = set()
        replacements = []
        for called_method, caller_locations in callee_methods.items():
            if called_method == method_key:
                continue
            if self.is_method_overload(called_method):
                continue
            definition, modified_called_method = self._find_callee(called_method, self.all_definitions, all_class_parents)
            if definition is None:
                continue
            
            sub_source = self._expand_method_source(root_caller_method, modified_called_method, all_calls, all_class_parents, depth + 1, max_depth)
            if sub_source is not None:
                callee_source, sub_callees = sub_source[0], sub_source[2][0]['callees']
            else:
                callee_source = definition['source']
                sub_callees = []
            callee_len = self._normalized_function_length(callee_source)
            if callee_len < 5:
                continue
            callee_file = self.get_file_from_module(modified_called_method[0])
            callee = {"source": callee_source, 'decorators': definition['decorators'], "file": callee_file}
            callee['position'] = {"module_path": modified_called_method[0], "class_name": modified_called_method[1], "method_name": modified_called_method[2]}
            callee_testsuites = self._find_related_testsuite(called_method)
            if len(callee_testsuites) > 3:
                callee_testsuites = random.choices(list(callee_testsuites), k=3)
            testunits.update(set(callee_testsuites))

            for caller_location in caller_locations:
                caller = caller_location
                caller['source'] = caller_method_definition['source']
                caller['position'] = {"module_path": method_key[0], "class_name": method_key[1], "method_name": method_key[2]}
                caller['file'] = caller_file
                caller['caller_start_line'] = caller_method_definition['start_line']
                caller_replacement = self.generate_replacement_caller_from_callee(caller, callee, all_class_parents)
                if caller_replacement is None:
                    continue
                if isinstance(caller_replacement['replacement'], str):
                    caller_replacement['replacement'] = caller_replacement['replacement'].splitlines()
                rel_start = caller_replacement['start'] - caller_method_definition['start_line'] + 1
                rel_end = caller_replacement['end'] - caller_method_definition['start_line'] + 1
                caller_replacement['rel_start'] = rel_start
                caller_replacement['rel_end'] = rel_end
                replacements.append(caller_replacement)
                imports.extend(caller_replacement['imports'])
                callees.append({"type": "callee", 
                                "decorators": definition['decorators'], 
                                "start": caller_replacement['start'], 
                                "end": caller_replacement['end'], 
                                "code": callee_source, 
                                "callees": sub_callees,
                                'position': callee['position']})
            if len(replacements) >= self.MINIMAL_CALLEE_NUM:
                if sub_source is not None:
                    _, _, sub_callee, _imports, _testunits = sub_source
                    imports.extend(_imports)
                    testunits.update(_testunits)
        expanded_source = caller_method_definition['source']
        if len(replacements) >= self.MINIMAL_CALLEE_NUM:
            method_lines = caller_method_definition['source'].splitlines()
            for replacement in sorted(replacements, key=lambda x: x['rel_start'], reverse=True):
                method_lines[replacement['rel_start']:replacement['rel_end']] = replacement['replacement']
            expanded_source = "\n".join(method_lines)
        after_caller_replacement = [{"type": "caller" if depth==0 else "callee", "code": expanded_source, "position": {"module_path": method_key[0], "class_name": method_key[1], "method_name": method_key[2]}, 'callees': callees}]
        
        self._expanded_method_cache[cache_key] = (expanded_source, replacements, after_caller_replacement, imports, testunits)

        return self._expanded_method_cache[cache_key]
    
    def collect_once(self, caller_method, all_class_parents, all_calls, max_depth=1):
        caller_module = caller_method[0]
        caller_file = self.get_file_from_module(caller_module)
        if not os.path.exists(caller_file):
            return
        # 统计该调用者调用的所有方法的总行数
        caller_method_definition = self.all_definitions.get(caller_method)
        if caller_method_definition is None:
            return
        caller_source = caller_method_definition['source']
        caller_len = self._normalized_function_length(caller_source)
        if caller_len < 5:
            return
        # after_refactor_code = [{"type": "caller", "code": caller_source, "position": {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}, 'callees': []}]
        caller_testsuites = self._find_related_testsuite(caller_method)
        if len(caller_testsuites) > 10:
            caller_testsuites = random.choices(list(caller_testsuites), k=10)
        testsuites = set(caller_testsuites)
        replacements = []

        caller_source = self._expand_method_source(caller_method, caller_method, all_calls, all_class_parents, depth=0, max_depth=max_depth)
        if caller_source is None:
            return
        
        _, caller_replacements, after_caller_replacement, imports, callee_testsuites = caller_source
        testsuites.update(callee_testsuites)
        replacements.extend(caller_replacements)
        after_refactor_code = after_caller_replacement
        # for called_method, caller_locations in callee_methods.items():
        #     if called_method == caller_method:
        #         continue
        #     # 获取被调用方法, 由于存在重载问题， 需要进行多次查找
        #     definition, modified_called_method = self._find_callee(called_method, self.all_definitions, all_class_parents)
        #     if definition is None:
        #         continue
        #     expanded_callee_source = self._expand_method_source(modified_called_method, all_calls, all_class_parents, depth=1)
        #     callee_source = expanded_callee_source if expanded_callee_source is not None else definition['source']
        #     callee_len = self._normalized_function_length(callee_source)
        #     if callee_len < 5:
        #         continue
        #     callee_testsuites = self._find_related_testsuite(called_method)
        #     if len(callee_testsuites) > 3:
        #         callee_testsuites = random.choices(list(callee_testsuites), k=3)
        #     testsuites.update(set(callee_testsuites))
        #     callee_file = self.get_file_from_module(modified_called_method[0])
        #     callee = {"source": callee_source, 'decorators': definition['decorators'], "file": callee_file }
        #     callee['position'] = {"module_path": modified_called_method[0], "class_name": modified_called_method[1], "method_name": modified_called_method[2]}
        #     for caller_location in caller_locations:
        #         caller = caller_location
        #         caller['source'] = caller_method_definition['source']
        #         caller['position'] = {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}
        #         caller['file'] = caller_file
        #         caller['caller_start_line'] = caller_method_definition['start_line']
        #         caller_replacement = self.generate_replacement_caller_from_callee(caller, callee, all_class_parents)
        #         if caller_replacement is None:
        #             continue
                    
        #         if isinstance(caller_replacement['replacement'], str):
        #             caller_replacement['replacement'] = caller_replacement['replacement'].splitlines()
        #         caller_replacement['lines'] = len(callee_source.splitlines())
        #         rel_start = caller_replacement['start'] - caller_method_definition['start_line'] + 1
        #         rel_end = caller_replacement['end'] - caller_method_definition['start_line'] + 1
        #         caller_replacement['rel_start'] = rel_start
        #         caller_replacement['rel_end'] = rel_end
        #         after_refactor_code[0]['callees'].append({"type": "callee", "decorators": definition['decorators'], "start": caller_replacement['start'], "end": caller_replacement['end'], "code": callee_source, 'position': callee['position']})
        #         replacements.append(caller_replacement)
        if len(replacements) < self.MINIMAL_CALLEE_NUM:
            return
        replacements[0]['imports'] = imports
        replacement_dict = {
            caller_method: replacements
        }
        total_callee_lines = sum([len(r['replacement']) for r in replacements])
        before_refactor_code, caller_lines = self.do_replacement(replacement_dict, caller_module)
        total_caller_lines = len(before_refactor_code[0]['code'].splitlines())
        # 如果被调用方法的总行数超过阈值
        if total_callee_lines > self.TOTAL_CALLEE_LENGTH_THRESHOLD:
            caller_file_content = [{"code": "\n".join(caller_lines), "module_path": caller_module, "file_suffix": caller_method[2]}]
            smell_content = self.create_diff_file(caller_file_content)
            long_method = {
                "type": self.name(),
                "meta":{"key": f"depth_{max_depth}", "depth": max_depth, "calling_times": len(replacements), "total_caller_lines": total_caller_lines, "total_callee_lines": total_callee_lines},
                "testsuites": list(testsuites),
                "after_refactor_code": after_refactor_code,
                "before_refactor_code": before_refactor_code,
                "smell_content": smell_content,
                "hash": hashcode("\n".join(caller_lines))
            }
            return long_method
    
    def collect(self, all_calls, all_class_parents):
        """
        @param class_methods: {(called_module, called_class, called_method_name): [(module_path, class_name, call_locations)]}
        @param all_calls: {(caller_method): {(called_method): [(module_path, class_name, call_locations)]}}
        @param all_definitions: {(module, class, method_name): definition}
        
        call_locations: { (called_module, called_class, called_method_name): [caller_infos]}
        @return: List of methods [{"type": LongMethod , "total_callee_lines": int, "after_refactor_code": after_refactor_code, "before_refactor_code": before_refactor_code, "caller_file_content": caller_file_content}]
        
        在本方法中，主要是以单个方法为起点，搜索在此方法中调用的其他方法，并将搜索到的方法内容填充到caller中，以达到扩充caller内容的目的。
        """
        
        long_methods = {}
        
        # 遍历所有调用者方法
        for caller_method, callee_methods in tqdm(all_calls.items(), desc="Scaning Long Method"):
            for max_depth in range(self.max_inline_depth):
                max_depth += 1
                long_method = self.collect_once(caller_method=caller_method, all_class_parents=all_class_parents, all_calls=all_calls, max_depth=max_depth)
                if long_method and long_method['hash'] not in long_methods:
                    long_methods[long_method['hash']] = long_method
        
        return list(long_methods.values())
