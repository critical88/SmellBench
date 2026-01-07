import ast
import os
import random

from base_method import BaseCollector
from collections import defaultdict
import textwrap
from utils import strip_python_comments


class DuplicatedMethodCollector(BaseCollector):
    def __init__(self, project_path, project_name, src_path) -> None:
        super().__init__(project_path, project_name, src_path)

    
    def collect(self, class_methods, all_calls, all_definitions, all_class_parents, family_classes):
        """
        @param class_methods: {(called_module, called_class, called_method_name): [(module_path, class_name, call_locations)]}
        @param all_calls: {(caller_method): {(called_method): [(module_path, class_name, call_locations)]}}
        @param all_definitions: {(module, class, method_name): definition}
        
        call_locations: { (called_module, called_class, called_method_name): [caller_infos]}
        @return: List of methods [{"type": DuplicatedMethod , "total_callee_lines": int, "after_refactor_code": after_refactor_code, "before_refactor_code": before_refactor_code, "caller_file_content": caller_file_content}]
        
        在本方法中，主要是以callee为起点，找到所有调用callee的方法，即caller，并将callee的内容填充到所有caller中，以达到重复代码的目的。
        """
        # 设置被调用次数的阈值（次数）
        TOTAL_CALLER_SIZE_THRESHOLD = 3
        # we discard the short callee, which is hard to recognize 
        CALLEE_MINIMAL_LEN = 5
        # we only preserve 10 callers for one callee to avoid unnecessary overhead
        MAX_CALLER_THRESHOLD = 10
        
        duplicated_methods = []
        # 遍历所有调用者方法
        for callee_method, caller_methods in class_methods.items():
            if len(caller_methods) < TOTAL_CALLER_SIZE_THRESHOLD:
                continue
            definition, modified_called_method = self._find_callee(callee_method, all_definitions, all_class_parents)
            callee_method = modified_called_method
            called_definition = definition
            if not called_definition or not called_definition.get('source'):
                continue
            callee_lines = self._normalized_function_length(called_definition['source'])
            # unnormalized_callee_lines = len(called_definition['source'].splitlines())
            if callee_lines < CALLEE_MINIMAL_LEN:
                continue
            callee_file = self.get_file_from_module(callee_method[0])
            callee = {"source": called_definition['source'], 'decorators': called_definition['decorators'], "file": callee_file }
            callee['position'] = {"module_path": callee_method[0], "class_name": callee_method[1], "method_name": callee_method[2]}

            before_refactor_code = []
            after_refactor_code = []
            caller_file_contents = []
            testsuites = set()
            valid_calling_times = 0
            
            caller_replacement_dict = defaultdict(dict)
            selected_replacements = defaultdict(dict)
            ## if one function is called over 10 times, then we randomly choose 10 callers to avoid resource waste. 
            all_callers = []
            for caller_composite in caller_methods:
                # ('urllib3.src.urllib3.contribopenssl', 'WrappedSocket', 'recv')
                caller_method = caller_composite['position']
                caller_module = caller_method[0]
                caller_locations = caller_composite['call_locations']
                caller_file = self.get_file_from_module(caller_module)
                
                if not os.path.exists(caller_file):
                    continue
                caller_method_definition = all_definitions.get(caller_method)
                
                replacements = []
                # 获取被调用方法的定义
                
                for caller_location in caller_locations:
                    caller = caller_location
                    caller['source'] = caller_method_definition['source']
                    caller['position'] = {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}
                    caller['file'] = caller_file
                    caller_replacement = self.generate_replacement_caller_from_callee(caller, callee, all_class_parents)
                    if caller_replacement is None:
                        continue
                        
                    if isinstance(caller_replacement['replacement'], str):
                        caller_replacement['replacement'] = caller_replacement['replacement'].splitlines()
                    caller_replacement['lines'] = len(called_definition['source'].splitlines())
                    rel_start = caller_replacement['start'] - caller_method_definition['start_line'] + 1
                    rel_end = caller_replacement['end'] - caller_method_definition['start_line'] + 1
                    caller_replacement['rel_start'] = rel_start
                    caller_replacement['rel_end'] = rel_end
                    
                    replacements.append(caller_replacement)
                if len(replacements) == 0:
                    continue
                caller_replacement_dict[caller_module][caller_method] = replacements
                all_callers.append(caller_method)
            ## only use selected refactor codes
            if len(all_callers) > MAX_CALLER_THRESHOLD:
                selected_callers = random.sample(all_callers, MAX_CALLER_THRESHOLD)
            else:
                selected_callers = all_callers
            for caller_method in selected_callers:
                caller_module = caller_method[0]
                caller_method_definition = all_definitions.get(caller_method)
                after_refactor = {"type": "caller", "code": caller_method_definition['source'], "position": {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}, 'callees': []}
                replacements = caller_replacement_dict[caller_module][caller_method]
                for replacement in replacements:
                    after_refactor['callees'].append({"type": "callee", "decorators": called_definition['decorators'], "start": replacement['start'], "end": replacement['end'], "code": called_definition['source'], 'position': callee['position']})
                after_refactor_code.append(after_refactor)
                selected_replacements[caller_module][caller_method] = replacements
                valid_calling_times += len(replacements)
                testsuites.update(self._find_related_testsuite(caller_method))

            for caller_module, caller_replacements in selected_replacements.items():

                before_refactors, caller_lines = self.do_replacement(caller_replacements, caller_module, all_definitions)
                before_refactor_code.extend(before_refactors)

                caller_file_contents.append({
                    "code": "\n".join(caller_lines), 
                    "module_path": caller_module, 
                    "file_suffix": f"dup_{callee_method[2]}"
                })
                
            if len(before_refactor_code) == 0:
                continue
            duplicated_methods.append({
                "type": "DuplicatedMethod",
                "meta":{"calling_times": valid_calling_times, "num_caller": len(selected_callers) , "callee_lines": callee_lines},
                "testsuites": list(testsuites),
                "after_refactor_code": after_refactor_code,
                "before_refactor_code": before_refactor_code,
                "caller_file_content": caller_file_contents
            })
        
        # 按被调用方法的总行数降序排序
        # long_methods.sort(key=lambda x: x["total_callee_lines"], reverse=True)
        
        return duplicated_methods
