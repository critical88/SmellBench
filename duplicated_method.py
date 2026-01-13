import ast
import os
import random

from base_method import BaseCollector
from collections import defaultdict
import textwrap
from utils import strip_python_comments
from tqdm import tqdm


class DuplicatedMethodCollector(BaseCollector):
    def __init__(self, project_path, project_name, src_path, all_definitions) -> None:
        super().__init__(project_path, project_name, src_path, all_definitions)

    def name(self):
        return "duplicated"

    def get_callee_mapping(self, all_calls):
        # 按callee调用组织方法调用信息 
        callee_mapping = defaultdict(list)
        ## all_calls的组织结构是，key=》caller，即谁调用了，value=》called_methods，即被调用的方法
        ## 这个called_methods可能不是当前class，甚至不是当前file内的，但要求必须是同一repo内的
        ## 整体来说，就是当前方法里调用了哪些方法
        ## 后面要做的就是reverse这个过程，即找到每个方法被调用的次数
        for caller, called_methods in all_calls.items():
            ## 非包内的方法不考虑， 或类名为空
            if not caller[0].startswith(self.module_name) or caller[1] is None:
                continue
            module_path, class_name, caller_method_name = caller
            
            # 统计方法调用次数
            for called_method, call_locations in called_methods.items():
                called_module, called_class, called_method_name = called_method
                callee_mapping[(called_module, called_class, called_method_name)].append({"position": (module_path, class_name, caller_method_name), "call_locations": call_locations})
        
        return callee_mapping
    def collect(self, all_calls, all_class_parents, family_classes):
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

        callee_mapping = self.get_callee_mapping(all_calls)
        
        duplicated_methods = []
        # 遍历所有调用者方法
        for callee_method, caller_methods in tqdm(callee_mapping.items(), desc="scanning duplicated codes"):
            if len(caller_methods) < TOTAL_CALLER_SIZE_THRESHOLD:
                continue
            definition, modified_called_method = self._find_callee(callee_method, self.all_definitions, all_class_parents)
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
            callee_testunits = self._find_related_testsuite(callee_method)
            if len(callee_testunits) > 10:
                callee_testunits = random.choices(list(callee_testunits), k=10)
            testsuites = set(callee_testunits)
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
                caller_method_definition = self.all_definitions.get(caller_method)
                
                replacements = []
                # 获取被调用方法的定义
                
                for caller_location in caller_locations:
                    caller = caller_location
                    caller['caller_start_line'] = caller_method_definition['start_line']
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
                caller_method_definition = self.all_definitions.get(caller_method)
                after_refactor = {"type": "caller", "code": caller_method_definition['source'], "position": {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}, 'callees': []}
                replacements = caller_replacement_dict[caller_module][caller_method]
                for replacement in replacements:
                    after_refactor['callees'].append({"type": "callee", "decorators": called_definition['decorators'], "start": replacement['start'], "end": replacement['end'], "code": called_definition['source'], 'position': callee['position']})
                after_refactor_code.append(after_refactor)
                selected_replacements[caller_module][caller_method] = replacements
                valid_calling_times += len(replacements)
                caller_testunits = self._find_related_testsuite(caller_method)
                if len(caller_testunits) > 3:
                    caller_testunits = random.choices(list(caller_testunits), k=3)
                testsuites.update(set(caller_testunits))

            for caller_module, caller_replacements in selected_replacements.items():

                before_refactors, caller_lines = self.do_replacement(caller_replacements, caller_module, self.all_definitions)
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
