import ast
import os

from base_method import BaseCollector
from tqdm import tqdm
import random
class LongMethodCollector(BaseCollector):

    def __init__(self, project_path, project_name, src_path, all_definitions) -> None:
        super().__init__(project_path, project_name, src_path, all_definitions)
    
    def name(self):
        return "long"
    
    def collect_once(self, caller_method, callee_methods, all_class_parents):
        MINIMAL_CALLEE_NUM = 2
        TOTAL_CALLEE_LENGTH_THRESHOLD = 20
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
        after_refactor_code = [{"type": "caller", "code": caller_source, "position": {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}, 'callees': []}]
        caller_testsuites = self._find_related_testsuite(caller_method)
        if len(caller_testsuites) > 10:
            caller_testsuites = random.choices(list(caller_testsuites), k=10)
        testsuites = set(caller_testsuites)
        replacements = []
        for called_method, caller_locations in callee_methods.items():
            if called_method == caller_method:
                continue
            # 获取被调用方法, 由于存在重载问题， 需要进行多次查找
            definition, modified_called_method = self._find_callee(called_method, self.all_definitions, all_class_parents)
            if definition is None:
                continue
            callee_source = definition['source']
            callee_len = self._normalized_function_length(callee_source)
            if callee_len < 5:
                continue
            callee_testsuites = self._find_related_testsuite(called_method)
            if len(callee_testsuites) > 3:
                callee_testsuites = random.choices(list(callee_testsuites), k=3)
            testsuites.update(set(callee_testsuites))
            callee_file = self.get_file_from_module(modified_called_method[0])
            callee = {"source": callee_source, 'decorators': definition['decorators'], "file": callee_file }
            callee['position'] = {"module_path": modified_called_method[0], "class_name": modified_called_method[1], "method_name": modified_called_method[2]}
            for caller_location in caller_locations:
                caller = caller_location
                caller['source'] = caller_method_definition['source']
                caller['position'] = {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}
                caller['file'] = caller_file
                caller['caller_start_line'] = caller_method_definition['start_line']
                caller_replacement = self.generate_replacement_caller_from_callee(caller, callee, all_class_parents)
                if caller_replacement is None:
                    continue
                    
                if isinstance(caller_replacement['replacement'], str):
                    caller_replacement['replacement'] = caller_replacement['replacement'].splitlines()
                caller_replacement['lines'] = len(definition['source'].splitlines())
                rel_start = caller_replacement['start'] - caller_method_definition['start_line'] + 1
                rel_end = caller_replacement['end'] - caller_method_definition['start_line'] + 1
                caller_replacement['rel_start'] = rel_start
                caller_replacement['rel_end'] = rel_end
                after_refactor_code[0]['callees'].append({"type": "callee", "decorators": definition['decorators'], "start": caller_replacement['start'], "end": caller_replacement['end'], "code": definition['source'], 'position': callee['position']})
                replacements.append(caller_replacement)
        if len(replacements) < MINIMAL_CALLEE_NUM:
            return
        replacement_dict = {
            caller_method: replacements
        }
        total_callee_lines = sum([len(r['replacement']) for r in replacements])
        before_refactor_code, caller_lines = self.do_replacement(replacement_dict, caller_module, self.all_definitions)
        total_caller_lines = len(before_refactor_code[0]['code'].splitlines())
        # 如果被调用方法的总行数超过阈值
        if total_callee_lines > TOTAL_CALLEE_LENGTH_THRESHOLD:
            
            long_method = {
                "type": "LongMethod",
                "meta":{"calling_times": len(replacements), "total_caller_lines": total_caller_lines, "total_callee_lines": total_callee_lines},
                "testsuites": list(testsuites),
                "after_refactor_code": after_refactor_code,
                "before_refactor_code": before_refactor_code,
                "caller_file_content": [{"code": "\n".join(caller_lines), "module_path": caller_module, "file_suffix": caller_method[2]}]
            }
            return long_method
    
    def collect(self, all_calls, all_class_parents, family_classes):
        """
        @param class_methods: {(called_module, called_class, called_method_name): [(module_path, class_name, call_locations)]}
        @param all_calls: {(caller_method): {(called_method): [(module_path, class_name, call_locations)]}}
        @param all_definitions: {(module, class, method_name): definition}
        
        call_locations: { (called_module, called_class, called_method_name): [caller_infos]}
        @return: List of methods [{"type": LongMethod , "total_callee_lines": int, "after_refactor_code": after_refactor_code, "before_refactor_code": before_refactor_code, "caller_file_content": caller_file_content}]
        
        在本方法中，主要是以单个方法为起点，搜索在此方法中调用的其他方法，并将搜索到的方法内容填充到caller中，以达到扩充caller内容的目的。
        """
        
        long_methods = []
        
        # 遍历所有调用者方法
        for caller_method, callee_methods in tqdm(all_calls.items(), desc="Scaning Long Method"):
            long_method = self.collect_once(caller_method=caller_method, callee_methods=callee_methods, all_class_parents=all_class_parents)
            if long_method:
                long_methods.append(long_method)
            
        
        return long_methods
