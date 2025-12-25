import ast
import os

from base_method import BaseCollector

class LongMethodCollector(BaseCollector):

    def __init__(self, project_path, project_name, src_path) -> None:
        super().__init__(project_path, project_name, src_path)
    
    
    def collect(self, class_methods, all_calls, all_definitions, all_class_parents, family_classes):
        """
        @param class_methods: {(called_module, called_class, called_method_name): [(module_path, class_name, call_locations)]}
        @param all_calls: {(caller_method): {(called_method): [(module_path, class_name, call_locations)]}}
        @param all_definitions: {(module, class, method_name): definition}
        
        call_locations: { (called_module, called_class, called_method_name): [caller_infos]}
        @return: List of methods [{"type": LongMethod , "total_callee_lines": int, "after_refactor_code": after_refactor_code, "before_refactor_code": before_refactor_code, "caller_file_content": caller_file_content}]
        
        在本方法中，主要是以单个方法为起点，搜索在此方法中调用的其他方法，并将搜索到的方法内容填充到caller中，以达到扩充caller内容的目的。
        """
        # 设置被调用方法总长度的阈值（行数）
        TOTAL_CALLEE_LENGTH_THRESHOLD = 20
        
        long_methods = []
        
        # 遍历所有调用者方法
        for caller_method, callee_methods in all_calls.items():
            caller_module = caller_method[0]
            caller_file = self.get_file_from_module(caller_module)
            # ('urllib3.src.urllib3.contribopenssl', 'WrappedSocket', 'recv')
            if not os.path.exists(caller_file):
                continue
            # 统计该调用者调用的所有方法的总行数
            before_refactor_code = []
            caller_method_definition = all_definitions.get(caller_method)
            after_refactor_code = [{"type": "caller", "code": caller_method_definition['source'], "position": {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}, 'callees': []}]
            replacements = []
            for called_method, caller_locations in callee_methods.items():
                # 获取被调用方法, 由于存在重载问题， 需要进行多次查找
                definition, modified_called_method = self._find_callee(called_method, all_definitions, all_class_parents)
                if definition is None:
                    continue
                callee_file = self.get_file_from_module(modified_called_method[0])
                callee = {"source": definition['source'], 'decorators': definition['decorators'], "file": callee_file }
                callee['position'] = {"module_path": modified_called_method[0], "class_name": modified_called_method[1], "method_name": modified_called_method[2]}
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
                    caller_replacement['lines'] = len(definition['source'].splitlines())
                    rel_start = caller_replacement['start'] - caller_method_definition['start_line'] + 1
                    rel_end = caller_replacement['end'] - caller_method_definition['start_line'] + 1
                    caller_replacement['rel_start'] = rel_start
                    caller_replacement['rel_end'] = rel_end
                    after_refactor_code[0]['callees'].append({"type": "callee", "decorators": definition['decorators'], "start": caller_replacement['start'], "end": caller_replacement['end'], "code": definition['source'], 'position': callee['position']})
                    replacements.append(caller_replacement)
            if len(replacements) == 0:
                continue
            replacement_dict = {
                caller_method: replacements
            }
            total_callee_lines = sum([len(r['replacement']) for r in replacements])
            before_refactor_code, caller_lines = self.do_replacement(replacement_dict, caller_module, all_definitions)
            total_caller_lines = len(before_refactor_code[0]['code'].splitlines())
            # 如果被调用方法的总行数超过阈值
            if total_callee_lines > TOTAL_CALLEE_LENGTH_THRESHOLD:
                testsuites = self._find_related_testsuite(caller_method)
                long_methods.append({
                    "type": "LongMethod",
                    "meta":{"calling_times": len(replacements), "total_caller_lines": total_caller_lines, "total_callee_lines": total_callee_lines},
                    "testsuites": list(testsuites),
                    "after_refactor_code": after_refactor_code,
                    "before_refactor_code": before_refactor_code,
                    "caller_file_content": [{"code": "\n".join(caller_lines), "module_path": caller_module, "file_suffix": caller_method[2]}]
                })
        
        # 按被调用方法的总行数降序排序
        # long_methods.sort(key=lambda x: x["total_callee_lines"], reverse=True)
        
        return long_methods
