import ast
import os

from base_method import BaseCollector

class LongMethodCollector(BaseCollector):

    def __init__(self, project_path, project_name, src_path) -> None:
        super().__init__(project_path, project_name, src_path)
    
    def get_file_from_module(self, module_path):

        return os.path.join(self.project_path, self.project_name, self.src_path, module_path.lstrip(self.project_name).lstrip(".").replace(".", os.sep) + ".py")

    def collect(self, class_methods, all_calls, all_definitions, all_caller_graph):
        """
        @param class_methods: {(called_module, called_class, called_method_name): [(module_path, class_name, call_locations)]}
        @param all_calls: {(caller_method): {(called_method): [(module_path, class_name, call_locations)]}}
        @param all_definitions: {(module, class, method_name): definition}
        
        call_locations: { (called_module, called_class, called_method_name): [caller_infos]}
        @return: List of methods [{"type": LongMethod , "total_callee_lines": int, "after_refactor_code": after_refactor_code, "before_refactor_code": before_refactor_code, "caller_file_content": caller_file_content}]
        
        在本方法中，主要是以单个方法为起点，搜索在此方法中调用的其他方法，并将搜索到的方法内容填充到caller中，以达到扩充caller内容的目的。
        """
        # 设置被调用方法总长度的阈值（行数）
        TOTAL_CALLEE_LENGTH_THRESHOLD = 50
        
        long_methods = []
        
        # 遍历所有调用者方法
        for caller_method, callee_methods in all_calls.items():
            caller_file = self.get_file_from_module(caller_method[0])
            # ('urllib3.src.urllib3.contribopenssl', 'WrappedSocket', 'recv')
            if not os.path.exists(caller_file):
                continue
            # 统计该调用者调用的所有方法的总行数
            before_refactor_code = []
            caller_method_definition = all_definitions.get(caller_method)
            after_refactor_code = [{"type": "caller", "code": caller_method_definition['source'], "position": {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}, 'callees': []}]
            replacements = []
            for called_method, caller_locations in callee_methods.items():
                # 获取被调用方法的定义
                definition = all_definitions.get(called_method)
                if not definition or not definition.get('source'):
                    continue
                callee_file = self.get_file_from_module(called_method[0])
                callee = {"source": definition['source'], 'decorators': definition['decorators'], "file": callee_file }
                callee['position'] = {"module_path": called_method[0], "class_name": called_method[1], "method_name": called_method[2]}
                for caller_location in caller_locations:
                    caller = caller_location
                    caller['source'] = caller_method_definition['source']
                    caller['position'] = {"module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}
                    caller['file'] = caller_file
                    caller_replacement = self.replace_caller_from_callee(caller, callee)
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
            caller_content, caller_tree = self._read_file(caller_file)
            # 分析调用者的文件
            caller_lines = caller_content.splitlines()
            caller_method_definition_lines = caller_method_definition['source'].splitlines()
            imports = []
            for replacement in sorted(replacements, key=lambda x: x['start'], reverse=True):
                caller_method_definition_lines[replacement['rel_start']:replacement['rel_end']] = replacement['replacement']
                imports.extend(replacement['imports'])
            total_callee_lines = len(caller_method_definition_lines)
            statements = self._convert_imports_to_statements(imports, caller_file)
            last_import_line = self._get_last_import_line(caller_file)
            # 先替换在后面的内容，防止序号错乱
            caller_start = caller_method_definition['start_line'] - 1
            
            if last_import_line < caller_method_definition['start_line']:
                caller_lines[caller_method_definition['start_line'] - 1: caller_method_definition['end_line']] = caller_method_definition_lines
                for stat in statements:
                    caller_lines.insert(last_import_line + 1, stat)
                caller_start += len(statements)
            else:
                for stat in statements:
                    caller_lines.insert(last_import_line + 1, stat)
                caller_lines[caller_method_definition['start_line'] - 1: caller_method_definition['end_line']] = caller_method_definition_lines
            caller_end = caller_start + total_callee_lines
            before_refactor_code = [{"type": "caller", "start": caller_start, "end": caller_end, "code": "\n".join(caller_method_definition_lines), "module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}]
            # 如果被调用方法的总行数超过阈值
            if total_callee_lines > TOTAL_CALLEE_LENGTH_THRESHOLD:
                testsuites = self._find_related_testsuite(caller_method, all_caller_graph)
                long_methods.append({
                    "type": "LongMethod",
                    "testsuites": list(testsuites),
                    "total_callee_lines": total_callee_lines,
                    "after_refactor_code": after_refactor_code,
                    "before_refactor_code": before_refactor_code,
                    "caller_file_content": [{"code": "\n".join(caller_lines), "module_path": caller_method[0], "method_name": caller_method[2]}]
                })
        
        # 按被调用方法的总行数降序排序
        long_methods.sort(key=lambda x: x["total_callee_lines"], reverse=True)
        
        return long_methods
