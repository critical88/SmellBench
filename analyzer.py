import ast
import os
from collections import defaultdict
from typing import Dict, Set, Tuple, List
from long_method import LongMethodCollector
from duplicated_method import DuplicatedMethodCollector

class MethodCallVisitor(ast.NodeVisitor):
    def __init__(self, project_path: str, project_name: str, src_path: str):
        self.current_class = None
        self.current_method = None
        # self.file_path = file_path
        self.project_path = project_path
        self.project_name = project_name
        self.src_path = src_path
        
        # 存储普通方法调用信息：{(模块路径, 类名, 方法名): {(模块路径, 类名, 被调用方法名)}}
        self.method_calls = defaultdict(dict)
        # 存储测试方法调用信息：与 method_calls 结构相同
        self.test_method_calls = defaultdict(dict)
        # 存储方法定义信息：{(模块路径, 类名, 方法名): 方法信息}
        self.method_definitions = {}

        self.caller_graph = defaultdict(list)
        
    def is_test_file(self, file_path: str) -> bool:
        """判断是否是测试文件"""
        file_name = os.path.basename(file_path).lower()
        return (
            file_name.startswith('test_') or
            file_name.endswith('_test.py') or
            'tests' in file_path.lower() or
            'test' in file_path.lower()
        )
        
    def is_test_method(self, method_name: str) -> bool:
        """判断是否是测试方法"""
        if not method_name:
            return False
        method_name = method_name.lower()
        return (
            method_name.startswith('test_') or
            'test' in method_name
        )
        
    def is_test_class(self, class_name: str) -> bool:
        """判断是否是测试类"""
        if not class_name:
            return False
        class_name = class_name.lower()
        return (
            class_name.startswith('test') or
            class_name.endswith('test') or
            'testcase' in class_name
        )
    
    def set_file_path(self, file_path, content):
        self.file_path = file_path
        self.module_path = os.path.relpath(file_path, self.project_path).replace(os.sep, '.').replace('.py', '')
        # 清空导入映射，因为每个文件的导入都是独立的
        self.imports = {}
        # 读取文件内容并保存原始行
        self.file_lines = content.splitlines()
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        # 记录类的完整定义信息
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Import(self, node):
        # 处理形如 import foo 或 import foo.bar as baz 的导入
        for alias in node.names:
            module_name = alias.name  # 完整的模块路径
            asname = alias.asname or module_name  # 如果有as就用as后的名字，否则用原名
            self.imports[asname] = module_name
            

    def visit_ImportFrom(self, node):
        # 处理形如 from foo.bar import baz 或 from . import baz 的导入
        if node.level > 0:  # 相对导入
            # 获取当前模块的部分
            parts = self.module_path.split('.')
            if node.level > len(parts):
                return  # 无效的相对导入
            # 移除适当数量的部分来解析相对导入
            base = '.'.join(parts[:-node.level])
            if node.module:  # from .foo import bar
                base = f"{base}.{node.module}" if base else node.module
        else:  # 绝对导入
            base = node.module or ''

        # 处理导入的名称
        for alias in node.names:
            if alias.name == '*':
                continue  # 暂时不处理 import *
            module_name = f"{base}.{alias.name}" if base else alias.name
            asname = alias.asname or alias.name
            self.imports[asname] = module_name 
        
    def is_typing_overload(self, node: ast.FunctionDef) -> bool:
        """检查函数是否带有@typing.overload装饰器"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'overload':
                return True
            if isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name) and decorator.value.id == 'typing' and decorator.attr == 'overload':
                    return True
        return False

    def visit_FunctionDef(self, node):
        method_key = (self.module_path, self.current_class, node.name)
        # 使用原始文件内容来保留格式
        original_lines = self.file_lines[node.lineno-1:node.end_lineno]
        original_source = '\n'.join(original_lines)
        self.method_definitions[method_key] = {
            'start_line': node.lineno,
            'end_line': node.end_lineno,
            'source': original_source,
            'file_path': self.file_path,
            'module_path': self.module_path,
            'decorators': [ast.unparse(d) for d in node.decorator_list]
        }
        # if self.current_class:
            # 忽略带有@typing.overload装饰器的方法
        if self.is_typing_overload(node):
            return
            
        old_method = self.current_method
        self.current_method = node.name
        
        # 存储方法定义信息，包含完整的模块路径
        self.generic_visit(node)
        self.current_method = old_method
    def module_exists(self, module_path):
        return os.path.exists(os.path.join(self.project_path, self.src_path, module_path.replace(".", os.sep) + ".py"))
        # return os.path.exists(os.path.join(self.project_path, module_path.replace(".", os.sep) + ".py"))
    
    def is_module_directory(self, module_path):
        return os.path.exists(os.path.join(self.project_path, module_path.replace(".", os.sep)))
    
    def is_camel_case(self, name):
        return name[0].isupper() and not all(c.isupper() for c in name) 
    def is_special_func(self, method_name):
        return method_name.startswith("__") and method_name.endswith("__")
    def visit_Call(self, node):
        if self.current_method is None:
            return
        caller = (self.module_path, self.current_class, self.current_method)

        # 判断是否是测试方法调用
        is_test = (
            self.is_test_file(self.file_path) or
            self.is_test_method(self.current_method) or
            self.is_test_class(self.current_class)
        )

        if isinstance(node.func, ast.Attribute):
            # 检查是否是super()调用
            if isinstance(node.func.value, ast.Call) and \
               isinstance(node.func.value.func, ast.Name) and \
               node.func.value.func.id == 'super':
                return  # 忽略super()调用
            
            called_method_name = node.func.attr
            if self.is_special_func(called_method_name):
                return
            
            # called_method = (self.module_path, self.current_class, called_method_name)
            
            
            # 处理两种情况：
            # 1. 类内部方法调用 (self.method)
            if isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'self' and \
               self.current_class:
                called_method = (self.module_path, self.current_class, called_method_name)
                if called_method not in self.method_definitions:
                    return 
                    # 记录调用位置信息
                call_info = {
                    'caller_method': self.current_method,
                    'line_number': node.lineno,
                    'col_offset': node.col_offset,
                    'end_line_number': node.end_lineno,
                    'end_col_offset': node.end_col_offset,
                    'source': "\n".join(self.file_lines[node.lineno-1:node.end_lineno])
                }
                
                # 根据是否是测试方法选择存储位置
                target_dict = self.test_method_calls if is_test else self.method_calls
                
                if caller not in target_dict:
                    target_dict[caller] = {}
                if called_method not in target_dict[caller]:
                    target_dict[caller][called_method] = []
                target_dict[caller][called_method].append(call_info)
                self.caller_graph[called_method].append((caller, is_test))
                
            
            # 2. 工具类方法调用 (module.method) 或类方法调用 (ClassName.method)
            inner_node = node.func
            name_id = []
            while hasattr(inner_node, 'value'):
                inner_node = inner_node.value
                if hasattr(inner_node, "attr"):
                    name_id.append(inner_node.attr)
            name_id.reverse()
            if isinstance(inner_node, ast.Name):
                # 获取调用者的模块名或类名
                name_id.insert(0, inner_node.id)
                name_id = ".".join(name_id) 
                # 先检查是否是导入的模块
                if name_id not in self.imports:
                    return
                module_path = self.imports[name_id]
                if not module_path.startswith(self.project_name):
                    return
                if self.module_exists(module_path):
                    called_method = (module_path, None, called_method_name)
                # TODO 暂不考虑目录的情况，例如(urllib3.src.util)里的util为目录，但是__init__.py里写了一些方法导致可以用util.xx()调用
                elif self.is_module_directory(module_path):
                    return
                else:
                    from_name = module_path.split(".")[-1]
                    module_path = ".".join(module_path.split(".")[:-1])
                    # 如果是called_method_name，表示这个方法是工具类方法，否则是类方法
                    if from_name == called_method_name:
                        called_method = (module_path, None, called_method_name)
                    else:
                        ## 检查是否符合驼峰命名法（至少包含一个大写字母）
                        if self.is_camel_case(from_name):
                            called_method = (module_path, from_name, called_method_name)
                        ## 不符合驼峰表示其代表一个对象的调用方法，例如dict.get()方法
                        else:
                            return
                
                # 记录调用位置信息
                call_info = {
                    'caller_method': self.current_method,
                    'line_number': node.lineno,
                    'col_offset': node.col_offset,
                    'end_line_number': node.end_lineno,
                    'end_col_offset': node.end_col_offset,
                    'source': "\n".join(self.file_lines[node.lineno-1:node.end_lineno])
                }
                
                # 根据是否是测试方法选择存储位置
                target_dict = self.test_method_calls if is_test else self.method_calls
                
                if caller not in target_dict:
                    target_dict[caller] = {}
                if called_method not in target_dict[caller]:
                    target_dict[caller][called_method] = []
                target_dict[caller][called_method].append(call_info)
                self.caller_graph[called_method].append((caller, is_test))
            
        self.generic_visit(node)
class MethodFilter():
    def __init__(self):
        pass

    def before(self, caller, called_methods):
        return True
    
    def filter(self, caller, called_methods):
        return True



class MethodAnalyzer():
    def __init__(self, project_name: str, src_path: str, project_path: str):
        self.project_path = project_path
        self.src_path = src_path
        self.project_name = project_name
        self.module_name = os.path.join(self.project_name, self.src_path).replace("/", ".").replace(os.sep, ".")
        self.package_root = os.path.dirname(project_path)
        self.visitor = MethodCallVisitor(project_path, project_name, src_path)


    def analyze_file(self, file_path) -> Tuple[Dict, Dict, Dict]:
        print(f"Analyzing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return {}, {}, {}
        visitor = self.visitor
        visitor.set_file_path(file_path, content)
        try:
            tree = ast.parse(content)
            visitor.visit(tree)
            return visitor.method_calls, visitor.test_method_calls, visitor.method_definitions, visitor.caller_graph
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}, {}, {}

    def collect_test_module(self, test_calls):
        """
        为每个方法找到对应的test方法，并存在test_suites_methods
        即使用method_key值就可以找到对应的test方法所在的module_name, class_name, method_name
        @return: test_suites_methods: Dict[Tuple[str, str, str], List[Tuple[str, str, str]]]
        """
        test_suites_methods = defaultdict(list)
        for caller, called_methods in test_calls.items():
            for called_method, _ in called_methods.items():
                test_suites_methods[called_method].append(caller)
        return test_suites_methods

    def _combine_graph(self, old_graph, new_graph):
        return_graph = {}
        for k, v in old_graph.items():
            if k in new_graph:
                return_graph[k] = v + new_graph[k]
            else:
                return_graph[k] = v
        for k, v in new_graph.items():
            if k not in return_graph:
                return_graph[k] = v
        return return_graph

    def find_refactor_codes(self) -> List[Tuple[Tuple[str, str, str], int, Dict]]:
        project_path = os.path.join(self.project_path, self.project_name)
        print(f"Starting analysis of directory: {project_path}")
        if not os.path.exists(project_path):
            print(f"Error: Directory {project_path} does not exist!")
            return []
        
        # 获取项目根目录（src目录的父目录）
        project_root = os.path.dirname(os.path.dirname(project_path))
        print(f"Project root: {project_root}")
        
        all_calls = {}
        all_definitions = {}
        all_test_calls = {}
        all_caller_graph = {}
        
        # 遍历所有Python文件
        for root, dirs, files in os.walk(project_path):
            src_path = os.path.join(self.project_path, self.project_name, self.src_path)
            if not os.path.normpath(root).startswith(os.path.normpath(src_path)) and \
                        not (root.__contains__("test")):
                continue
            # 不检索.xxx的信息
            if any([_dir.startswith(".") and not _dir.startswith("..") for _dir in root.split(os.sep)]):
                continue
            print(f"\nScanning directory: {root}")
            print(f"Found {len(files)} files")
            
            for file in files:
                if file.endswith('.py'):
                    try:
                        calls, test_calls, definitions, caller_graph = self.analyze_file(os.path.join(root, file))
                        all_calls.update(calls)
                        all_definitions.update(definitions)
                        all_test_calls.update(test_calls)
                        all_caller_graph = self._combine_graph(all_caller_graph, caller_graph)
                        print(f"Found {len(calls)} normal calls, {len(test_calls)} test calls, and {len(definitions)} definitions in {file}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        traceback.print_exc()


        # 按类组织方法调用信息 
        class_methods = defaultdict(list)
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
                class_methods[(called_module, called_class, called_method_name)].append((module_path, class_name, caller_method_name, call_locations))
        test_suites_methods = self.collect_test_module(all_test_calls)
        result = []
        collectors = [
            LongMethodCollector(self.project_path, self.project_name, self.src_path), 
            # DuplicatedMethodCollector(self.project_path, self.project_name, self.src_path)
        ]
        for collector in collectors:
            ret = collector.collect(class_methods, all_calls, all_definitions, all_caller_graph)
            filtered_ret = []
            for r in ret:
                cover_all_caller_test = True
                # test_suites = set()
                # for code in r['before_refactor_code']:
                #     ## 如果所有的caller全在test_suites_methods中，就说明了测试全覆盖。
                #     if code['type'] == 'caller' and \
                #      (code['module_path'], code['class_name'], code['method_name']) in test_suites_methods:
                #         test_suites.add((code['module_path']))
                #     else:
                #         cover_all_caller_test = False
                #         break
                # if not cover_all_caller_test:
                #     continue
                # r['test_suites'] = list(test_suites)
                filtered_ret.append(r)
            result.extend(filtered_ret)
        print(f"number of refactor_codes: {len(result)}")
        return result