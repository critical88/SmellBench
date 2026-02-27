import ast
import os
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional
from long_method import LongMethodCollector
from duplicated_method import DuplicatedMethodCollector
from overloaded_method import OverloadedMethodCollector
from base_method import BaseCollector
import json
import traceback
import subprocess
from utils import pushd, _log, DEBUG_LOG_LEVEL
from pathlib import Path

class MethodCallVisitor(ast.NodeVisitor):
    def __init__(self, 
                 project_path: str, 
                 project_name: str, 
                 src_path: str, 
                 all_classes: Set[Tuple], 
                 class_parent: Dict[Tuple, List[Tuple]],
                 function_variables: Dict[str, Dict[str, List[str]]]):
        self.current_class = None
        self.current_method = None
        # self.file_path = file_path
        self.project_path = project_path
        self.project_name = project_name
        self.module_name = os.path.normpath(src_path).split(os.sep)[-1]
        self.src_path = src_path
        
        # 存储普通方法调用信息：{(模块路径, 类名, 方法名): {(模块路径, 类名, 被调用方法名)}}
        self.method_calls = defaultdict(dict)
        # 存储测试方法调用信息：与 method_calls 结构相同
        self.test_method_calls = defaultdict(dict)
        # 存储方法定义信息：{(模块路径, 类名, 方法名): 方法信息}
        self.method_definitions = {}
        self.all_classes = all_classes
        self.class_parent = class_parent
        self.function_variables = function_variables
        self.method_scope = []
        self._init_vars()

    
    def _init_vars(self):
        self.class_attr_types = defaultdict(dict)
        self.class_methods = defaultdict(set)
        self.variable_types_stack = []
        self.global_var_types = {}
        self.local_class_names = set()


    def is_test_file(self, file_path: str) -> bool:
        """判断是否是测试文件"""
        file_name = os.path.basename(file_path).lower()
        return (
            file_name.startswith('test_') or
            file_name.endswith('_test.py') or
            'tests' in file_path.lower() or
            'test' in file_path.lower()
        )
        
    def is_test_method(self, method: tuple) -> bool:
        """判断是否是测试方法"""
        if not method:
            return False
        method_name = method[2].lower()
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
    
    def set_file_path(self, file_path: str, content:str):
        file_path = os.path.normpath(file_path)
        self.file_path = file_path
        src_prefix = os.path.normpath(os.path.join(self.project_path, self.project_name, self.src_path))
        if file_path.startswith(src_prefix):
            rel_path = os.path.relpath(file_path, src_prefix)
        else:
            rel_path = os.path.relpath(file_path, os.path.join(self.project_path, self.project_name))
        self.module_path = self.module_name + "." + rel_path.replace(os.sep, '.').replace('.py', '')
        # 清空导入映射，因为每个文件的导入都是独立的
        self.imports: Dict[str, str]= {}
        # 读取文件内容并保存原始行
        self.file_lines = content.splitlines()
        self._init_vars()
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.local_class_names.add(node.name)
        if node.name not in self.class_attr_types:
            self.class_attr_types[node.name] = {}
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
    
    def _bind_method_variables(self, module, class_name, method_name):
        func = ""
        if class_name is not None:
            func += class_name + "."
        func += method_name
        key = f"{module}:{func}"
        if key not in self.function_variables:
            return 
        variables = self.function_variables[key]
        for vars, types in variables.items():
            if len(types) == 0:
                continue
            module = ".".join(types[0].split(".")[:-1]) 
            _cls = types[0].split(".")[-1]
            if (module, _cls) in self.all_classes:
                self._bind_target_type(vars, (module, _cls))
    def _bind_variables(self):
        """
        first bind the __init__ func of ancestors
        then bind the __init__ func of current class
        finally bind the variables in current method
        """
        # bind __init__ func of ancestors
        if self.current_class is not None:
            parent_key = (self.module_path, self.current_class)
            if parent_key in self.class_parent:
                ancestors = self.class_parent[parent_key]
                while len(ancestors) > 0:
                    cur_parent = ancestors.pop()
                    if cur_parent in self.class_parent:
                        ancestors += self.class_parent[cur_parent]
                    self._bind_method_variables(cur_parent[0], cur_parent[1], "__init__")
            # then bind the __init__ func of current class
            self._bind_method_variables(self.module_path, self.current_class, "__init__")
        ## finally bind the variables in current method
        self._bind_method_variables(self.module_path, self.current_class, self.current_method)


    def visit_FunctionDef(self, node):
        method_name = self.method_scope + [node.name]
        method_name = ".".join(method_name)
        
        method_key = (self.module_path, self.current_class, method_name)
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
        ## add current file methods
        if self.current_class is None:
            self.imports[method_name] = self.module_path

        if self.current_class:
            class_key = (self.module_path, self.current_class)
            self.class_methods[class_key].add(method_name)
            
        old_method = self.current_method
        self.current_method = method_name
        
        # 存储方法定义信息，包含完整的模块路径
        self.variable_types_stack.append({})
        self._bind_variables()
        self.method_scope.append(node.name)
        self.generic_visit(node)
        self.method_scope.pop()
        self.variable_types_stack.pop()
        self.current_method = old_method
    def module_exists(self, module_path):
        prefix_path = Path(self.project_path) / self.project_name / self.src_path
        prefix_path = prefix_path.parent
        _path = prefix_path / (module_path.replace(".", os.sep) + ".py")
        return os.path.exists(_path)
        
    def is_module_directory(self, module_path):
        return os.path.exists(os.path.join(self.project_path, module_path.replace(".", os.sep)))
    
    def is_class(self, module_name, class_name):
        return (module_name, class_name) in self.all_classes
    def is_camel_case(self, name: str):
        if name.startswith("_"):
            name = name.lstrip("_")
        return name[0].isupper() and not all(c.isupper() for c in name) 
    def is_special_func(self, method_name: str):
        return method_name.startswith("__") and method_name.endswith("__")
    def visit_Call(self, node):
        if self.current_method is None or self.is_special_func(self.current_method):
            return
        caller = (self.module_path, self.current_class, self.current_method)

        # 判断是否是测试方法调用
        is_test = (
            self.is_test_file(self.file_path) or
            self.is_test_method(caller) or
            self.is_test_class(self.current_class)
        )
        is_test_method = self.is_test_method(caller)
        if isinstance(node.func, ast.Name):
            called_method_name = node.func.id
            if self._is_constructor_call(node.func):
                return
            if self.is_special_func(called_method_name):
                return
            # 检查是否是类方法调用
            if self.current_class and called_method_name in self.class_attr_types[self.current_class]:
                called_method = (self.module_path, self.current_class, called_method_name)
                self._store_call(caller, called_method, node, is_test)
                return
            # 检查是否是工具类方法调用
            if called_method_name in self.imports:
                module_path = self.imports[called_method_name]
                if not module_path.startswith(self.module_name):
                    return
                if self.module_exists(module_path):
                    called_method = (module_path, None, called_method_name)
                elif self.is_module_directory(module_path):
                    return
                else:
                    from_name = module_path.split(".")[-1]
                    module_path = ".".join(module_path.split(".")[:-1])
                    if from_name == called_method_name:
                        called_method = (module_path, None, called_method_name)
                    else:
                        if self.is_class(module_path, from_name):
                            called_method = (module_path, from_name, called_method_name)
                        else:
                            return
                self._store_call(caller, called_method, node, is_test)
                return

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
            handled_call = False
            if isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'self' and \
               self.current_class:
                called_method = (self.module_path, self.current_class, called_method_name)
                if called_method not in self.method_definitions:
                    return 
                ### self.method，此处的method可能会根据不同的object而变化，需要进一步确认
                self._store_call(caller, called_method, node, is_test)
                handled_call = True
            ## object.method()调用
            else:
                resolved = self._resolve_object_class(node.func.value)
                if resolved:
                    module_path, class_name = resolved
                    called_method = (module_path, class_name, called_method_name)
                    self._store_call(caller, called_method, node, is_test)
                    handled_call = True
            
            # 2. 工具类方法调用 (module.method) 或类方法调用 (ClassName.method)
            if not handled_call:
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
                    if not module_path.startswith(self.module_name):
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
                            if self.is_class(module_path, from_name):
                                called_method = (module_path, from_name, called_method_name)
                            ## 不符合驼峰表示其代表一个对象的调用方法，例如dict.get()方法
                            else:
                                return
                    self._store_call(caller, called_method, node, is_test)
            
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        value_type = self._infer_type_from_value(node.value)
        if value_type:
            for target in node.targets:
                self._bind_target_type(target, value_type)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        value_type = self._infer_type_from_value(node.value) if node.value else None
        if not value_type:
            value_type = self._infer_type_from_annotation(node.annotation)
        if value_type:
            self._bind_target_type(node.target, value_type)
        self.generic_visit(node)

    def _store_call(self, caller, called_method, node, is_test):
        if not called_method:
            return
        call_info = {
            'caller_method': self.current_method,
            'line_number': node.lineno,
            'col_offset': node.col_offset,
            'end_line_number': node.end_lineno,
            'end_col_offset': node.end_col_offset,
            'source': "\n".join(self.file_lines[node.lineno-1:node.end_lineno])
        }
        target_dict = self.test_method_calls if is_test else self.method_calls
        if caller not in target_dict:
            target_dict[caller] = {}
        if called_method not in target_dict[caller]:
            target_dict[caller][called_method] = []
        target_dict[caller][called_method].append(call_info)
        

    def _bind_target_type(self, target, value_type):
        if not value_type:
            return
        if isinstance(target, ast.Name) or isinstance(target, str):
            if isinstance(target, ast.Name):
                target = target.id
            if self.variable_types_stack:
                self.variable_types_stack[-1][target] = value_type
            else:
                self.global_var_types[target] = value_type
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and \
               target.value.id == 'self' and \
               self.current_class:
                self.class_attr_types.setdefault(self.current_class, {})[target.attr] = value_type

    def _infer_type_from_value(self, value):
        if value is None:
            return None
        if isinstance(value, ast.Call):
            return self._resolve_class_from_name(value.func)
        if isinstance(value, ast.Name):
            return self._lookup_var_type(value.id)
        if isinstance(value, ast.Attribute):
            return self._resolve_object_class(value)
        return None

    def _infer_type_from_annotation(self, annotation):
        if annotation is None:
            return None
        if isinstance(annotation, ast.Subscript):
            return self._infer_type_from_annotation(annotation.value)
        return self._resolve_class_from_name(annotation)

    def _lookup_var_type(self, name):
        for scope in reversed(self.variable_types_stack):
            if name in scope:
                return scope[name]
        return self.global_var_types.get(name)

    def _resolve_object_class(self, node):
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self._lookup_var_type(node.id)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and \
               node.value.id == 'self' and \
               self.current_class:
                return self.class_attr_types.get(self.current_class, {}).get(node.attr)
            return None
        return None

    def _get_name_chain(self, node):
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            base = self._get_name_chain(node.value)
            if base is None:
                return None
            return base + [node.attr]
        if isinstance(node, ast.Subscript):
            return self._get_name_chain(node.value)
        return None

    def _resolve_class_from_name(self, node):
        name_chain = self._get_name_chain(node)
        if not name_chain:
            return None
        if len(name_chain) == 1 and name_chain[0] in self.local_class_names:
            return self.module_path, name_chain[0]
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
                if module_path and class_name and module_path.startswith(self.module_name):
                    if not self.is_class(module_path, class_name):
                        return None
                    return module_path, class_name
                return None
        return None

    def _is_constructor_call(self, func_node):
        return self._resolve_class_from_name(func_node) is not None

    def _split_module_and_class(self, full_path):
        if not full_path or "." not in full_path:
            return None, None
        parts = full_path.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        if not module_path or not class_name:
            return None, None
        return module_path, class_name
class MethodFilter():
    def __init__(self):
        pass

    def before(self, caller, called_methods):
        return True
    
    def filter(self, caller, called_methods):
        return True
    
class MethodAnalyzer():
    def __init__(self, project_name: str,  project_path: str, long_method_depth: Optional[int] = 1):
        
        meta_info, all_classes, self.class_parent, self.function_testunit, function_variables = self._read_meta_info(project_name)
        self.meta_info = meta_info
        src_path = meta_info['src_path']
        self.project_path = project_path
        self.src_path = src_path
        self.project_name = project_name
        self.module_name = os.path.normpath(src_path).split(os.sep)[-1]
        self.package_root = os.path.dirname(project_path)
        self.visitor = MethodCallVisitor(project_path, project_name, src_path, all_classes, self.class_parent, function_variables)
        self.long_method_depth = long_method_depth

    def _read_meta_info(self, project_name):
        testunit_file = os.path.join("output", project_name, f"function_testunit_mapping.json")
        if not os.path.exists(testunit_file):
            raise Exception("please first run `testunit_coverage.py` to generate mapping file.")
        with open(testunit_file, 'r', encoding='utf-8') as f:
            meta_info = json.load(f)
        if len(meta_info['functions']) == 0:
            raise Exception("failed to generate mapping file, please check the log and fix the bug then rerun it")
        all_classes = set()
        all_class_parent = defaultdict(list)
        for _, _cls in meta_info['classes'].items():
            key = (_cls['module'], _cls['qualname'])
            all_classes.add(key)
            for module, parent_cls  in _cls['bases']:
                all_class_parent[key].append((module, parent_cls))
        
        functions_info = meta_info['functions']
        function_testunit = {}
        function_variables = {}
        for k, v in functions_info.items():
            function_testunit[k] = v['tests']
            function_variables[k] = v['variable_types']
            
        return meta_info['meta'], all_classes, all_class_parent, function_testunit, function_variables

    def analyze_file(self, file_path) -> Tuple[Dict, Dict, Dict]:
        _log(f"Analyzing file: {file_path}")
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
            return visitor.method_calls, visitor.method_definitions
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}, {}, {}
        
    def _collect_family_classes(self, all_class_parent):
        """
        Return each class's inheritance family as sets of related classes.
        """
        family_classes = defaultdict(set)

        class_graph = defaultdict(set)

        for class_key, parents in all_class_parent.items():
            class_graph.setdefault(class_key, set())
            for parent_key in parents:
                class_graph.setdefault(parent_key, set())
                class_graph[class_key].add(parent_key)
                class_graph[parent_key].add(class_key)

        visited = set()
        for class_key in class_graph.keys():
            if class_key in visited:
                continue
            component = set()
            stack = [class_key]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in class_graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            for comp_class in component:
                family_classes[comp_class].update(component)

        return family_classes
    

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
        all_class_parents = self.class_parent
        
        # 遍历所有Python文件
        for root, dirs, files in os.walk(project_path):
            src_path = os.path.join(self.project_path, self.project_name, self.src_path)
            if not os.path.normpath(root).startswith(os.path.normpath(src_path)):
                continue
            if any( p.startswith(".") or (p.startswith("__") and not p=='__init__.py') for p in os.path.relpath(root, project_path).split(os.sep)):
                continue
            ignored_packages = ["venv", "site-packages", "test"]
            if any(ignored in root for ignored in ignored_packages):
                continue
            
            for file in files:
                if file.endswith('.py'):
                    try:
                        calls, definitions = self.analyze_file(os.path.join(root, file))
                        all_calls.update(calls)
                        all_definitions.update(definitions)
                        _log(f"Found {len(calls)} normal calls, and {len(definitions)} definitions in {file}", level=DEBUG_LOG_LEVEL)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        traceback.print_exc()

        family_classes = self._collect_family_classes(all_class_parents)

        with pushd(os.path.join(self.project_path, self.project_name)):
            commit_hash = subprocess.run(['git', "rev-parse", "HEAD"], text=True, cwd=".", capture_output=True, check=True).stdout.strip()
        
        collectors: List[BaseCollector] = [
            LongMethodCollector(project_path=self.project_path, 
                                project_name=self.project_name, 
                                src_path=self.src_path, 
                                all_definitions=all_definitions, 
                                family_classes=family_classes, 
                                commitid=commit_hash,
                                max_inline_depth=self.long_method_depth),
            DuplicatedMethodCollector(project_path=self.project_path, 
                                    project_name=self.project_name, 
                                    src_path=self.src_path, 
                                    commitid=commit_hash,
                                    all_definitions=all_definitions, 
                                    family_classes=family_classes),
            # OverloadedMethodCollector(self.project_path, self.project_name, self.src_path)
        ]
        result = {}
        refactored_count = 0
        refactor_codes = []
        
        stat = {}
        for collector in collectors:
            ret = collector.collect(all_calls=all_calls, 
                                    all_class_parents=all_class_parents)
            refactored_count += len(ret)
            filtered_ret = []
            stat_ret = defaultdict(int)
            for r in ret:
                if len(r['testsuites']) > 0:
                    r['commit_hash'] = commit_hash
                    r['project_name'] = self.project_name
                    filtered_ret.append(r)
                    if "key" in r['meta']:
                        key = r['meta']['key']
                        stat_ret[key] += 1
            stat[collector.name()] = {
                "total": len(filtered_ret),
                **stat_ret
            }
            refactor_codes.extend(filtered_ret)
        
        result['refactor_codes'] = refactor_codes
        result['stat'] = {
            "split": stat,
            "raw_refacoter_num": refactored_count,
            "refactor_with_test_num": len(refactor_codes)
        }
        return result

