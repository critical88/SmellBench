import ast
import json
import os
import re
from client import LLMFactory
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Set
import textwrap
from utils import strip_python_comments, _log, DEBUG_LOG_LEVEL


class BaseCollector():
    def __init__(self, project_path:str, project_name:str, src_path:str, all_definitions:Dict[Tuple, Dict], family_classes) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        :param project_path: Description
        :type project_path: str
        :param project_name: Description
        :type project_name: str
        :param src_path: Description
        :type src_path: str
        :param all_definitions: the method definition of the repo
            example: 
            {(called_module, called_class, called_method_name): definition}
        
        """
        self.project_path = project_path
        self.project_name = project_name
        self.module_name = os.path.normpath(src_path).split(os.sep)[-1]
        self.src_path = src_path
        self.file_cache = {}
        self.all_definitions = all_definitions
        self.family_classes = family_classes
        self.language = "python"
        self._init_llm_client()
        self._init_testunit()

    def name(self):
        raise NotImplementedError()
    
    def is_method_overload(self, method_key: Tuple) -> bool:
        module_name, class_name, method_name = method_key
        if class_name is None:
            return False
        
        for family_class in self.family_classes.get((module_name, class_name), set()):
            if family_class[1] == class_name and family_class[0] == module_name:
                continue
            if (family_class[0], family_class[1], method_name) in self.all_definitions:
                return True
        
        return False

    def _init_testunit(self):
        testunit_file = os.path.join("output", self.project_name, f"function_testunit_mapping.json")
        if not os.path.exists(testunit_file):
            raise Exception("please first run `testunit_cover.py` to generate mapping file.")
        with open(testunit_file, 'r', encoding='utf-8') as f:
            self._function_testunit = json.load(f)['functions']
    
    def _find_callee(self, called_method, all_definitions, all_class_parents):
        definition = all_definitions.get(called_method)
        parents = []
        cur_class = (called_method[0], called_method[1])
        if cur_class in all_class_parents:
            parents = all_class_parents[cur_class]
        cache_parents = set()
        while not definition and len(parents) > 0:
            cur_class = parents.pop()
            if cur_class in cache_parents:
                continue
            called_method = (cur_class[0], cur_class[1], called_method[2])
            definition = all_definitions.get((cur_class[0], cur_class[1], called_method[2]))
            cache_parents.add(cur_class)
            if cur_class in all_class_parents:
                parents.extend(all_class_parents[cur_class])
        if definition is None or not definition.get('source'):
            return None, None
        called_method = (cur_class[0], cur_class[1], called_method[2])
        return definition, called_method
    
    def get_file_from_module(self, module_path: str):

        return os.path.join(self.project_path, self.project_name, self.src_path, module_path.lstrip(self.module_name).lstrip(".").replace(".", os.sep) + ".py")
    
    def _safe_replace_identifier(self, source: str, old_name: str, replacement: str) -> str:
        if not old_name:
            return source
        if old_name == replacement:
            return source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            pattern = r'\b' + re.escape(old_name) + r'\b'
            return re.sub(pattern, replacement, source)

        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        spans = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == old_name:
                parent = parents.get(node)
                if isinstance(parent, ast.Attribute) and parent.attr == old_name and parent.value is not node:
                    continue
                start_line = getattr(node, 'lineno', None)
                start_col = getattr(node, 'col_offset', None)
                end_line = getattr(node, 'end_lineno', None)
                end_col = getattr(node, 'end_col_offset', None)
                if start_line is None or start_col is None:
                    continue
                if end_line is None or end_col is None:
                    end_line = start_line
                    end_col = start_col + len(old_name)
                spans.append(((start_line, start_col), (end_line, end_col)))

        if not spans:
            return source

        lines = source.splitlines(keepends=True)
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        def to_offset(pos):
            line, col = pos
            return line_offsets[line - 1] + col

        replacements = []
        for start, end in sorted(spans, key=lambda p: (p[0][0], p[0][1])):
            replacements.append((to_offset(start), to_offset(end)))

        if not replacements:
            return source

        result_parts = []
        last_index = 0
        for start, end in replacements:
            result_parts.append(source[last_index:start])
            result_parts.append(replacement)
            last_index = end
        result_parts.append(source[last_index:])
        return ''.join(result_parts)

    def _generate_unique_name(self, base_name: str, existing_names) -> str:
        candidate = base_name
        counter = 0
        while candidate in existing_names:
            counter += 1
            candidate = f"{base_name}_{counter}"
        existing_names.add(candidate)
        return candidate

    def _fallback_return_rewrite(self, body_source: str, return_var: str | None) -> str:
        pattern = r'\breturn\s+([^\n;]+)'
        if return_var:
            return re.sub(pattern, f'{return_var} = \\1', body_source)
        return body_source

    def _build_return_nodes(self, node: ast.Return, flag_names: list[str] | None, return_var: str | None):
        new_nodes = []
        if return_var:
            if node.value:
                value_code = ast.unparse(node.value)
                assign_code = f"{return_var} = {value_code}"
            else:
                assign_code = f"{return_var} = None"
            new_nodes.append(ast.parse(assign_code).body[0])
        else:
            new_nodes.append(ast.Pass())

        if flag_names:
            for flag_name in flag_names:
                flag_assign = ast.Assign(
                    targets=[ast.Name(id=flag_name, ctx=ast.Store())],
                    value=ast.Constant(value=True)
                )
                new_nodes.append(flag_assign)
        return new_nodes

    def _stmt_contains_return(self, stmt):
        if isinstance(stmt, ast.Return):
            return True
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
        if isinstance(stmt, ast.If):
            return any(self._stmt_contains_return(s) for s in stmt.body + stmt.orelse)
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            return any(self._stmt_contains_return(s) for s in stmt.body + stmt.orelse)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            return any(self._stmt_contains_return(s) for s in stmt.body)
        if isinstance(stmt, ast.Try):
            if any(self._stmt_contains_return(s) for s in stmt.body + stmt.finalbody + stmt.orelse):
                return True
            for handler in stmt.handlers:
                if any(self._stmt_contains_return(s) for s in handler.body):
                    return True
            return False
        if hasattr(ast, "Match") and isinstance(stmt, ast.Match):
            return any(self._stmt_contains_return(s) for case in stmt.cases for s in case.body)
        return False

    def _block_contains_return(self, statements):
        for stmt in statements:
            if self._stmt_contains_return(stmt):
                return True
        return False

    def _block_has_direct_return(self, statements):
        for stmt in statements:
            if isinstance(stmt, ast.Return):
                return True
        return False

    def _process_statement(self, stmt, return_var: str | None, existing_names, loop_flag=None, extra_flags=None, in_loop_body=False):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return [stmt], None
        if isinstance(stmt, ast.Return):
            flag_names = []
            if loop_flag:
                flag_names.append(loop_flag)
            if extra_flags:
                flag_names.extend(extra_flags)
            nodes = self._build_return_nodes(stmt, flag_names if flag_names else None, return_var)
            info = {'has_return': True, 'type': 'return'}
            if loop_flag and in_loop_body:
                nodes.append(ast.Break())
                info['loop_flag'] = loop_flag
            return nodes, info
        if isinstance(stmt, ast.If):
            contains_return = self._stmt_contains_return(stmt)
            body_has_direct_return = self._block_has_direct_return(stmt.body)
            needs_if_flag = contains_return and not body_has_direct_return
            flag_assign = None
            flag_name = None
            body_flags = extra_flags
            if needs_if_flag:
                flag_name = self._generate_unique_name("__inline_if_flag", existing_names)
                existing_names.add(flag_name)
                flag_assign = ast.Assign(
                    targets=[ast.Name(id=flag_name, ctx=ast.Store())],
                    value=ast.Constant(value=False)
                )
                if extra_flags:
                    body_flags = extra_flags + [flag_name]
                else:
                    body_flags = [flag_name]
            stmt.body, body_info = self._process_block_with_returns(stmt.body, return_var, existing_names, loop_flag, body_flags, in_loop_body=in_loop_body)
            stmt.orelse, orelse_info = self._process_block_with_returns(stmt.orelse, return_var, existing_names, loop_flag, body_flags if needs_if_flag else extra_flags, in_loop_body=in_loop_body)
            has_return = (body_info and body_info.get('has_return')) or (orelse_info and orelse_info.get('has_return'))
            info = None
            if has_return:
                info = {'has_return': True, 'type': 'if', 'node': stmt}
                if needs_if_flag:
                    info['uses_flag'] = True
                    info['flag_name'] = flag_name
            statements = [stmt]
            if flag_assign:
                statements = [flag_assign, stmt]
            return statements, info
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            needs_flag = self._block_contains_return(stmt.body + stmt.orelse)
            loop_flag_name = loop_flag if needs_flag else None
            flag_assign = None
            if needs_flag and loop_flag_name is None:
                loop_flag_name = self._generate_unique_name("__inline_loop_flag", existing_names)
                existing_names.add(loop_flag_name)
                flag_assign = ast.Assign(
                    targets=[ast.Name(id=loop_flag_name, ctx=ast.Store())],
                    value=ast.Constant(value=False)
                )
            stmt.body, body_info = self._process_block_with_returns(stmt.body, return_var, existing_names, loop_flag_name, extra_flags, in_loop_body=True)
            stmt.orelse, orelse_info = self._process_block_with_returns(stmt.orelse, return_var, existing_names, loop_flag_name, extra_flags, in_loop_body=False)
            has_return = (body_info and body_info.get('has_return')) or (orelse_info and orelse_info.get('has_return'))
            info = {'has_return': True, 'type': 'loop', 'flag_name': loop_flag_name} if has_return else None
            statements = [stmt]
            if flag_assign:
                statements = [flag_assign, stmt]
            return statements, info
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt.body, body_info = self._process_block_with_returns(stmt.body, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
            has_return = body_info and body_info.get('has_return')
            info = {'has_return': True} if has_return else None
            return [stmt], info
        if isinstance(stmt, ast.Try):
            stmt.body, body_info = self._process_block_with_returns(stmt.body, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
            stmt.finalbody, final_info = self._process_block_with_returns(stmt.finalbody, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
            stmt.orelse, orelse_info = self._process_block_with_returns(stmt.orelse, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
            handler_return = False
            for handler in stmt.handlers:
                handler.body, handler_info = self._process_block_with_returns(handler.body, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
                if handler_info and handler_info.get('has_return'):
                    handler_return = True
            has_return = (body_info and body_info.get('has_return')) or \
                         (final_info and final_info.get('has_return')) or \
                         (orelse_info and orelse_info.get('has_return')) or handler_return
            info = {'has_return': True, 'type': 'try'} if has_return else None
            return [stmt], info
        if hasattr(ast, "Match") and isinstance(stmt, ast.Match):
            case_return = False
            for case in stmt.cases:
                case.body, case_info = self._process_block_with_returns(case.body, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
                if case_info and case_info.get('has_return'):
                    case_return = True
            info = {'has_return': True, 'type': 'match'} if case_return else None
            return [stmt], info
        return [stmt], None
    def _normalized_function_length(self, source: str) -> int:
        text = strip_python_comments(source)
        normalized_lines = [
                line for line in text.splitlines() if line.strip()
        ]
        return len(normalized_lines)
    def _process_block_with_returns(self, statements, return_var: str | None, existing_names, loop_flag=None, extra_flags=None, in_loop_body=False):
        new_statements = []
        idx = 0
        while idx < len(statements):
            stmt = statements[idx]
            transformed, info = self._process_statement(stmt, return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body)
            new_statements.extend(transformed)
            if info and info.get('has_return'):
                remaining, _ = self._process_block_with_returns(
                    statements[idx + 1:], return_var, existing_names, loop_flag, extra_flags, in_loop_body=in_loop_body
                )
                if info.get('type') == 'if':
                    if info.get('uses_flag'):
                        flag_name = info.get('flag_name')
                        if remaining:
                            guard_if = ast.If(
                                test=ast.UnaryOp(
                                    op=ast.Not(),
                                    operand=ast.Name(id=flag_name, ctx=ast.Load())
                                ),
                                body=remaining,
                                orelse=[]
                            )
                            new_statements.append(guard_if)
                    else:
                        if remaining:
                            info['node'].orelse.extend(remaining)
                    return new_statements, {'has_return': True}
                elif info.get('type') == 'loop':
                    flag_name = info.get('flag_name')
                    if remaining:
                        if flag_name:
                            guard_if = ast.If(
                                test=ast.UnaryOp(
                                    op=ast.Not(),
                                    operand=ast.Name(id=flag_name, ctx=ast.Load())
                                ),
                                body=remaining,
                                orelse=[]
                            )
                            new_statements.append(guard_if)
                        else:
                            new_statements.extend(remaining)
                    return new_statements, {'has_return': True}
                else:
                    return new_statements, {'has_return': True}
            idx += 1
        return new_statements, None

    def _rewrite_returns_with_flag(self, body_source: str, return_var: str | None, existing_names) -> str:
        try:
            module = ast.parse(body_source)
        except SyntaxError:
            return self._fallback_return_rewrite(body_source, return_var)

        has_return_stmt = any(isinstance(node, ast.Return) for node in ast.walk(module))
        if not has_return_stmt:
            return body_source

        processed_body, _ = self._process_block_with_returns(module.body, return_var, existing_names)

        new_module = ast.Module(body=processed_body, type_ignores=[])
        new_module = ast.fix_missing_locations(new_module)
        return "\n".join(ast.unparse(stmt) for stmt in new_module.body)

    def _init_llm_client(self):
        self.client = LLMFactory.create_client()

    def _find_related_testsuite(self, current_methods)->Set:
        if not current_methods:
            return set()
        key = current_methods[0]
        if current_methods[1] is None:
            key += f":{current_methods[2]}"
        else:
            key += f":{current_methods[1]}.{current_methods[2]}"
        
        if key in self._function_testunit:
            return set(self._function_testunit[key]['tests'])
        
        return set()


    def _read_file(self, file_path: str) -> tuple:
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        with open(file_path, 'r', encoding='utf-8') as f:
            file_contents = f.read()
        file_tree = ast.parse(file_contents)
        self.file_cache[file_path] = (file_contents, file_tree)
        return file_contents, file_tree


    
    def _get_module_info(self, module_path: str, current_file: str, level: int = 0) -> tuple:
        """
        解析模块路径，返回 (source, absolute_path)
        Args:
            module_path: 模块路径（可能是相对路径）
            current_file: 当前文件的路径
            level: 相对导入的层级（点的数量）
        Returns:
            (来源类型, 绝对路径)
        """
        if level == 0 and not module_path.startswith('.'):
            return 'third_party', module_path
        else:
            # 相对导入
            current_dir = os.path.dirname(current_file)
            # 根据点的数量往上进目录
            for _ in range(level - 1):
                current_dir = os.path.dirname(current_dir)
            
            if module_path.startswith('.'):
                module_path = module_path[1:]
            
            if module_path:
                abs_path = os.path.join(current_dir, module_path.replace(".", os.sep) + ".py")
            else:
                abs_path = current_dir + ".py"
            
            # 检查是否在项目路径内
            if abs_path.startswith(self.project_path):
                return 'project', abs_path
            else:
                return 'third_party', module_path

    def _get_imports(self, file_path: str) -> dict:
        """
        获取文件中的所有导入语句和定义的符号
        返回一个字典，其中键是别名或原始名称，值是一个字典包含以下信息：
        - source: 来源（'third_party', 'project' 或 'local')
        - path: 如果是项目内部包，则是绝对路径，否则是原始导入路径
        - method: 导入或定义的内容（方法、类名或常量）

        例如：
        import numpy as np -> {'np': {'source': 'third_party', 'path': 'numpy', 'method': None}}
        class MyClass -> {'MyClass': {'source': 'local', 'path': '/current/file.py', 'method': 'MyClass'}}
        def my_func -> {'my_func': {'source': 'local', 'path': '/current/file.py', 'method': 'my_func'}}
        """
        imports = {}
        try:
            file_contents, tree = self._read_file(file_path)
            
            # 先收集文件中的所有一级定义
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    imports[node.name] = {
                        'source': 'local',
                        'path': file_path,
                        'method': node.name,
                    }
                elif isinstance(node, ast.FunctionDef):
                    imports[node.name] = {
                        'source': 'local',
                        'path': file_path,
                        'method': node.name,
                    }
                elif isinstance(node, ast.Assign):
                    # 只处理一级变量定义
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            imports[target.id] = {
                                'source': 'local',
                                'path': file_path,
                                'method': target.id,
                            }
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        target = node.target
                        imports[target.id] = {
                            'source': 'local',
                            'path': file_path,
                            'method': target.id,
                        }
            
            # 然后处理导入语句
            parent_map = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parent_map[child] = parent

            def is_inside_method(target_node):
                current = parent_map.get(target_node)
                while current:
                    if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return True
                    current = parent_map.get(current)
                return False

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if is_inside_method(node):
                        continue
                    for name in node.names:
                        module_path = name.name
                        key = name.asname if name.asname else module_path.split(".")[-1]
                        source, path = self._get_module_info(module_path, file_path)
                        imports[key] = {
                            'source': source,
                            'path': path,
                            'method': None,
                        }
                
                elif isinstance(node, ast.ImportFrom):
                    if is_inside_method(node):
                        continue
                    module = node.module if node.module else ''
                    source, path = self._get_module_info(module, file_path, node.level)

                    for name in node.names:
                        if name.name == '*':
                            imports[f'{module}.*'] = {
                                'source': source,
                                'path': path,
                                'method': '*',
                            }
                        else:
                            key = name.asname if name.asname else name.name.split(".")[-1]
                            imports[key] = {
                                'source': source,
                                'path': path,
                                'method': name.name,
                            }
        except Exception as e:
            print(f"Error analyzing imports in {file_path}: {e}")
        return imports

    def _analyze_callee_imports(self, callee_source: str) -> set:
        """分析被调用方法中使用的包和方法"""
        used_imports = set()
        try:
            tree = ast.parse(callee_source.strip())
            
            # 收集所有使用的名称
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_imports.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # 处理形如 module.function 的调用
                    parts = []
                    current = node
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                        used_imports.add('.'.join(reversed(parts)))
        except Exception as e:
            print(f"Error analyzing callee imports: {e}")
        return used_imports
    
    def _convert_imports_to_statements(self, imports, caller_file):
        """
        @param imports: [(import_name, {source: source, path: path, method: method})]
        @return statements: list(str), lineno
        将导入的包和方法转换为import语句
        并返回对应的语句
        """
        # 按照文件路径和引用分组导入
        imports_by_path = {}
        for imp in imports:
            alias, info = imp
            path_key = info['path']
            
            if info['source'] in ['local', 'project']:
                # 如果是本地定义，需要计算相对路径
                caller_dir = os.path.dirname(caller_file)
                callee_path = info['path']
                
                # 计算共同路径
                common_prefix = os.path.commonpath([caller_dir, callee_path])
                
                # 计算需要往上走多少层
                rel_caller = os.path.relpath(caller_dir, common_prefix)
                if rel_caller == ".":
                    # callee在caller的子目录中
                    dots = '.'
                else:
                    # 计算caller到common_prefix的层级数
                    dots = '.' * (len(rel_caller.split(os.sep)) + 1)
                
                # 计算callee相对于common_prefix的路径
                rel_callee = os.path.relpath(callee_path, common_prefix)
                # 移除.py后缀并转换为点分隔形式
                module_path = os.path.splitext(rel_callee)[0].replace(os.sep, '.')
                path_key = f"{dots}{module_path}"
            
            if path_key not in imports_by_path:
                imports_by_path[path_key] = []
            imports_by_path[path_key].append((alias, info))
        
        # 生成import语句
        import_statements = []
        for path_key, imports_group in imports_by_path.items():
            # 生成导入语句
            methods = set()
            for alias, info in imports_group:
                if info['method'] is None:
                    import_statements.append(f"import {path_key}")
                elif info['method'] == alias:
                    methods.add(f"{alias}")
                else:
                    methods.add(f"{info['method']} as {alias}")
            if len(methods) > 0:
                import_statements.append(f"from {path_key} import {', '.join(methods)}")
        return import_statements

    def _get_last_import_line(self, caller_file):
        """
        此方法要拿到caller_file下的最后一个import语句的行号
        @param caller_file: str, 文件路径
        @return: int, 最后一个import语句的行号，如果没有import语句则返回0
        """
        caller_content, caller_tree = self._read_file(caller_file)
        caller_lines = caller_content.splitlines()
        # find all import statements
        last_import_line = 0
        lines_without_import = 0
        
        # we can't insert import statements inside docstring
        docstring = False
        # some file starts with docstring, so we need to skip the docstring part
        first_docstring = False
        mutli_import = False
        for lineno, line in enumerate(caller_lines):
            ## if the line contains inline comment, we need to ignore the comment part
            if not line.strip().startswith("#") and "#" in line:
                line = line[:line.index("#")]
            if line.strip().startswith('"""'):
                docstring = not docstring
                if lineno < 3:
                    first_docstring = True
            if line.strip().endswith('""""') and line.strip() != '"""':
                docstring = not docstring
                first_docstring = False
            if docstring:
                if not first_docstring:
                    lines_without_import += 1
                    if lines_without_import > 3:
                        break
                continue
            if line.strip().startswith("from") and line.strip().endswith('('):
                lines_without_import = 0
                mutli_import = True
            if mutli_import and line.strip().endswith(")"):
                mutli_import = False
                last_import_line = lineno
            if mutli_import:
                continue
            if line.startswith("import") or line.strip().startswith("from"):
                last_import_line = lineno
                lines_without_import = 0
            elif line and not line.startswith('#'):
                # 只计算非空且非注释行
                lines_without_import += 1
                if lines_without_import > 3:
                    break

        return last_import_line

    def _get_imports_from_callee(self, caller, callee) -> list:
        if os.path.normpath(caller['file']) == os.path.normpath(callee['file']):
            return []
        # 1. 分析导入依赖
        caller_imports = self._get_imports(caller['file'])
        callee_imports = self._get_imports(callee['file'])
        callee_used_imports = self._analyze_callee_imports(callee['source'])
        
        ## filter imports
        filtered_callee_imports = {}
        for imp, info in callee_imports.items():
            if info['source'] == 'project':
                path = info['path']
                project_path = os.path.normpath(os.path.join(self.project_path, self.project_name, self.src_path, ".."))
                rel_path = os.path.relpath(os.path.normpath(path[:-3]), project_path)
                module_path = ".".join(rel_path.split(os.sep))
                if module_path == caller['position']['module_path']:
                    continue
            filtered_callee_imports[imp] = info
        # 找出需要添加的导入
        needed_imports = []
        
        for imp in callee_used_imports:
            # 检查是否在callee文件中定义或导入
            # if not any(imp.startswith(ci) for ci in caller_imports.keys()):
            if imp not in caller_imports:
                if imp in filtered_callee_imports:
                    info = filtered_callee_imports[imp]
                    needed_imports.append((imp, info))
        
        return needed_imports

    def generate_replacement_caller_from_callee(self, caller, callee, all_class_parents):
        ## ignore subfunction test.a means def test(): def a(): pass
        if "." in caller['position']['method_name']:
            return None
        if caller['caller_start_line'] > caller['line_number']:
            return None
        if caller['position'] == callee['position']:
            return None
        ret = self._generate_replace_caller_from_callee(caller, callee, all_class_parents)
        return ret
    
    def rearrange_caller_replacement(self, replacements_dict: Dict[Tuple, Dict])->Tuple[List[Dict], List]:
        imports = []
        caller_replacements = []
        for caller_method, replacements in replacements_dict.items():
            caller_method_definition = self.all_definitions[caller_method]
            caller_method_definition_lines = caller_method_definition['source'].splitlines()
            modified_len = 0
            for replacement in sorted(replacements, key=lambda x: x['start'], reverse=True):
                caller_method_definition_lines[replacement['rel_start']:replacement['rel_end']] = replacement['replacement']
                imports.extend(replacement['imports'])
                modified_len += len(replacement['replacement']) - (replacement['rel_end'] - replacement['rel_start'])
            
            caller_replacements.append({
                "caller_start": caller_method_definition['start_line'], 
                "caller_method": caller_method,
                "caller_method_definition": caller_method_definition,
                "modified_method_lines": caller_method_definition_lines,
                "modified_len": modified_len,
                })
        
        return caller_replacements, imports

    def _do_caller_replacement(self, caller_replacements: List[Dict], imports:List, caller_module):
        caller_file = self.get_file_from_module(caller_module)
        caller_content, _ = self._read_file(caller_file)
        # 分析调用者的文件
        caller_lines = caller_content.splitlines()
        before_refactor_code = []

        last_import_line = self._get_last_import_line(caller_file)
        statements = self._convert_imports_to_statements(imports, caller_file)

        sum_modified_len = sum([x['modified_len'] for x in caller_replacements])

        sum_modified_len += len(statements)
        ## assume import statements always locate the very first of the files.
        ## thus we first replace the caller_replacement and then insert import statement
        for caller_replacement in sorted(caller_replacements, key=lambda x: x['caller_start'], reverse=True):
            caller_method_definition = caller_replacement['caller_method_definition']
            caller_method_definition_lines = caller_replacement['modified_method_lines']
            sum_modified_len = sum_modified_len - caller_replacement['modified_len']
            caller_method = caller_replacement['caller_method']
            
            caller_lines[caller_method_definition['start_line'] - 1: caller_method_definition['end_line']] = caller_method_definition_lines
            
            caller_method_len = len(caller_method_definition_lines)
            
            caller_start = caller_method_definition['start_line'] - 1 + sum_modified_len
            caller_end = caller_start + caller_method_len
            
            before_refactor = {"type": "caller", "start": caller_start, "end": caller_end, "code": "\n".join(caller_method_definition_lines), "module_path": caller_method[0], "class_name": caller_method[1], "method_name": caller_method[2]}
            before_refactor_code.append(before_refactor)

        
        for stat in statements:
            caller_lines.insert(last_import_line + 1, stat)
        
        return before_refactor_code, caller_lines
    def do_replacement(self, replacements_dict: Dict[Tuple, Dict], caller_module):
        
        caller_replacements, imports = self.rearrange_caller_replacement(replacements_dict)
        
        return self._do_caller_replacement(caller_replacements, imports, caller_module)
        
    def _replace_caller_from_callee_gpt(self, caller, callee):
        ## 处理Import的问题
        imports = self._get_imports_from_callee(caller, callee)
         # 找到调用点
        start_line = caller['line_number'] - 1  # 转为0-based索引
        end_line = caller['end_line_number']
        
        # 分析调用者的文件
        caller_file = caller['file']

        caller_content, caller_tree = self._read_file(caller_file)
        
        caller_lines = caller_content.splitlines()
        original_call = caller_lines[start_line:end_line]
        
        # 分析调用语句，获取实际参数
        # 将多行调用合并成一行
        call_line = "\n".join(original_call).strip()
        # 解析整个表达式
        try:
            expr_ast = ast.parse(call_line)
        except Exception as e:
            print(f"Error parsing call: {e}")
            return None
        ## 如果是元组，说明是不合法的调用，需要特殊处理
        if isinstance(expr_ast.body[0], ast.Expr) and isinstance(expr_ast.body[0].value, ast.Tuple):
            return None
        
        # 分析调用语句，获取实际参数
        # 将多行调用合并成一行
        call_line = "\n".join(original_call).strip()
        decorators = "\n".join("@"+d for d in callee.get("decorators", []))
        
        prompt = """You are an expert in Python programming. You are expected to replace the caller statement with the callee source code.
Then output the replacement code in caller function. Note we only need the replacement code, not the whole caller function.
You are supposed to obey the following rules:
1. local variables in callee function should be carefully checked and replaced in caller function. if found conflicts with caller function, you should rename the variable in callee function.
2. return statements must be removed in callee function, if have multipe return statements, you should use IF ELSE to rearrange the callee function. Further, if have return value, you should carefully check and replace the return value in caller function.
3. ensure the replacement code is valid and can be compiled.
4. Think step by step and finally response the final revised code in ```python ```.

## Input
Caller Function:
```python
{caller}
```
Caller Line:
```python
{call_line}
```
Callee:
```python
{decorators}
{callee}
```
## Response
""".format(
    caller=caller['source'],
    call_line=call_line,
    decorators=decorators,
    callee=callee['source']
)
        # 调用GPT获取替换信息
        response = self.client.chat(
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=4096,
            temperature=0.7
        )
        
        content = response
        # 提取被```python```包裹的代码
        match = re.search(r'```python\s*([\s\S]*?)```', content)
        if match:
            python_code = match.group(1)
        else:
            return None
        python_code_lines = python_code.splitlines()

        start_indent = len(python_code_lines[0]) - len(python_code_lines[0].lstrip())
        
        indent = len(original_call[0]) - len(original_call[0].lstrip())
        indented_body = '\n'.join([' ' * (indent + len(line) - len(line.lstrip()) - start_indent) + line.lstrip() for line in python_code_lines])
        # 收集caller中的替换信息
        return {
            'start': start_line,
            'end': end_line,
            'replacement': indented_body.splitlines(),
            'imports': imports
        }
    def __get_args_from_ast(self, method_ast):
        args_info = {
            'args': [],  # 位置参数
            'defaults': [],  # 默认值
            'kwonly_args': [],  # 仅关键字参数
            'kwonly_defaults': [],  # 仅关键字参数的默认值
            'varargs': None,  # *args参数名
            'varkw': None  # **kwargs参数名
        }

        # 获取位置参数
        for arg in method_ast.args.args:
            # if arg.arg != 'self':
            args_info['args'].append(arg.arg)
        
        # 获取*args参数
        if method_ast.args.vararg:
            args_info['varargs'] = method_ast.args.vararg.arg
        
        # 获取**kwargs参数
        if method_ast.args.kwarg:
            args_info['varkw'] = method_ast.args.kwarg.arg
        
        # 获取默认值
        defaults = [ast.unparse(default) if default else 'None' for default in (method_ast.args.defaults or [])]
        args_info['defaults'] = ['None'] * (len(args_info['args']) - len(defaults)) + defaults
        
        # 获取仅关键字参数
        args_info['kwonly_args'] = [arg.arg for arg in method_ast.args.kwonlyargs]
        args_info['kwonly_defaults'] = [ast.unparse(default) if default else 'None' 
                                    for default in (method_ast.args.kw_defaults or [])]

        return args_info
        


    def _generate_replace_caller_from_callee(self, caller, callee, all_class_parents):
        """
        caller:调用者，{"file": caller_file, "caller_method": caller_method, "line_number": line_number, "col_offset": col_offset, "end_line_number": end_line_number}
        callee:被调用者, {"source": source, "file": file, "position": {"module_path": module, "class_name": class_name, "method_name": method_name}}
        本方法要完成的功能是：
        将callee的方法体插入到caller对应的调用位置上，
        处理变量冲突和包导入问题。
        """
        ## 处理Import的问题
        imports = self._get_imports_from_callee(caller, callee)

        ### deal callee
        # 获取方法的AST
        try:
            callee_method_ast = ast.parse(textwrap.dedent(callee['source'])).body[0]
        except:
            return
        decorators = callee.get("decorators")
        # 检查方法类型
        is_staticmethod = False
        is_classmethod = False
        if 'staticmethod' in decorators:
            is_staticmethod = True
        if 'classmethod' in decorators:
            is_classmethod = True
        if isinstance(callee_method_ast, ast.Pass):
            return
        args_info = self.__get_args_from_ast(callee_method_ast)
        args_info['is_staticmethod'] = is_staticmethod
        args_info['is_classmethod'] = is_classmethod

        # 分析方法体中的局部变量
        local_vars = set()
        for node in ast.walk(ast.parse(ast.unparse(callee_method_ast.body))):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_vars.add(node.id)
        
        _log(f"\nMethod information:", level=DEBUG_LOG_LEVEL)
        _log(f"Position arguments: {list(zip(args_info['args'], args_info['defaults']))}", level=DEBUG_LOG_LEVEL)
        _log(f"Keyword-only arguments: {list(zip(args_info['kwonly_args'], args_info['kwonly_defaults']))}", level=DEBUG_LOG_LEVEL)
        if args_info['varargs']:
            _log(f"Varargs (*{args_info['varargs']})", level=DEBUG_LOG_LEVEL)
        if args_info['varkw']:
            _log(f"Varkw (**{args_info['varkw']})", level=DEBUG_LOG_LEVEL)
        _log(f"Local variables: {local_vars}", level=DEBUG_LOG_LEVEL)
        
        # 移除装饰器、函数定义行和docstring，只保留函数体
        body_without_docstring = [node for node in callee_method_ast.body if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Str)]
        callee_method_body = ast.unparse(body_without_docstring)
        
        # 分析调用者的文件
        caller_file = caller['file']

        caller_content, caller_tree = self._read_file(caller_file)
        
        caller_lines = caller_content.splitlines()


        _log(f"\n---", level=DEBUG_LOG_LEVEL)
        _log(f"In method: {caller['caller_method']}", level=DEBUG_LOG_LEVEL)
        _log(f"At line {caller['line_number']}, column {caller['col_offset']}", level=DEBUG_LOG_LEVEL)
        # 分析caller方法中的局部变量
        caller_locals = set()
        caller_source_before_call = caller_lines[caller['caller_start_line']-1:caller['line_number']-1]
        last_statement = caller_lines[caller['line_number']-1]
        last_indent = len(last_statement) - len(last_statement.lstrip())
        caller_source_before_call += [' ' * last_indent + "pass"]
        try:
            caller_method_ast = ast.parse(textwrap.dedent("\n".join(caller_source_before_call)))
        except Exception as e:
            _log(f"Error parsing call: {e}", level=DEBUG_LOG_LEVEL)
            return
        if isinstance(caller_method_ast.body[0], ast.Pass):
            return
        caller_args_info = self.__get_args_from_ast(caller_method_ast.body[0])
        caller_locals = set(caller_args_info['args'])
        # 找到调用者的方法
        for node in ast.walk(caller_method_ast):
            # 只收集在当前行之前的变量
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                caller_locals.add(node.id)
        # special use for `self` 
        if "self" in caller_locals:
            caller_locals.remove("self")
            
        # 找到调用点
        start_line = caller['line_number'] - 1  # 转为0-based索引
        end_line = caller['end_line_number']
        
        original_call = caller_lines[start_line:end_line]
    
        # 分析调用语句，获取实际参数
        # 将多行调用合并成一行
        call_line = textwrap.dedent("\n".join(original_call))
        # 解析整个表达式
        try:
            expr_ast = ast.parse(call_line)
        except Exception as e:
            _log(f"Error parsing call: {e}", level=DEBUG_LOG_LEVEL)
            return
        first_node = expr_ast.body[0]
        ## 如果是元组，说明是不合法的调用，需要特殊处理
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Tuple):
            return
        # 检查是否是赋值语句
        if isinstance(first_node, ast.Assign):
            if len(first_node.targets) > 1:
                return
                
            target = first_node.targets[0]
            # 获取赋值语句左侧的变量名
            if isinstance(target, ast.Name):
                # 简单变量赋值，如 x = func()
                return_var = target.id
            else:
                # 属性赋值，如 self.x = func()
                return_var = ast.unparse(target)
            call_ast = first_node.value
        else:
            return_var = None
            call_ast = first_node.value if isinstance(first_node, ast.Expr) else None
            
        if not isinstance(call_ast, ast.Call):
            # 该调用可能只是某些方法的入参，或者计算表达式的一部分，因此无需考虑
            return
        called_name = None
        if isinstance(call_ast.func, ast.Attribute):
            called_name = call_ast.func.attr
        elif isinstance(call_ast.func, ast.Name):
            called_name = call_ast.func.id
        else:
            ## maybe the consecutive calling, such as `help_option(*help_option_names)(self)``
            return 
        if called_name != callee['position']['method_name']:
            ## this indicates the first calling method is not the callee in this line,
            ## e.g., add_data(self.parse_tuple(with_condexpr=True))
            ## first calling_method = parse_tuple but call_ast.func.id = add_data
            return
        
        # 获取位置参数
        actual_args = [ast.unparse(arg) for arg in call_ast.args]
        # 获取关键字参数
        actual_kwargs = {kw.arg: ast.unparse(kw.value) for kw in call_ast.keywords}
        _log(f"\nCall arguments:", level=DEBUG_LOG_LEVEL)
        _log(f"Position args: {actual_args}", level=DEBUG_LOG_LEVEL)
        _log(f"Keyword args: {actual_kwargs}", level=DEBUG_LOG_LEVEL)
        
        # 创建参数映射
        arg_mapping = {}
        
        # 处理位置参数
        min_args = len(args_info['args']) - len(args_info['defaults'])
        
        # 检查是否提供了足够的位置参数
        if len(actual_args) < min_args and not args_info['varargs']:
            raise ValueError(f"Not enough positional arguments. Expected at least {min_args}, got {len(actual_args)}")
        special_position = args_info['is_classmethod'] or (len(args_info['args']) > 0 and args_info['args'][0] == 'self')
        # 处理普通位置参数
        for i, arg_name in enumerate(args_info['args']):
            # 如果是第一个参数，需要特殊处理
            
            if special_position and i == 0:
                if args_info['is_classmethod']:
                    # 类方法的第一个参数是cls
                    if isinstance(call_ast.func, ast.Attribute):
                        # 如果是类方法调用 (如 ClassName.method())
                        class_name = ast.unparse(call_ast.func.value)
                        arg_mapping[arg_name] = class_name
                    elif i < len(actual_args):
                        # 如果通过位置参数传入了cls
                        arg_mapping[arg_name] = actual_args[i]
                    else:
                        # 如果通过关键字参数传入了cls
                        arg_mapping[arg_name] = actual_kwargs.get(arg_name, callee['position']['class_name'])
                    special_position = True
                ## some define parameter name as 'self'
                elif arg_name == 'self' and not isinstance(call_ast.func, ast.Name):
                    arg_mapping[arg_name] = ast.unparse(call_ast.func.value)
                else:
                    # 普通实例方法，第一个参数是self
                    if i < len(actual_args):
                        arg_mapping[arg_name] = actual_args[i]
                    elif arg_name in actual_kwargs:
                        arg_mapping[arg_name] = actual_kwargs[arg_name]
                    else:
                        arg_mapping[arg_name] = 'self'
            else:
                # 处理其他参数，需要考虑参数偏移
                # 对于实例方法和类方法，实际参数索引需要-1，因为第一个参数(self/cls)已经被特殊处理
                arg_index = i
                if special_position:
                    arg_index = i - 1
                
                if arg_index < len(actual_args):
                    arg_mapping[arg_name] = actual_args[arg_index]
                elif arg_name in actual_kwargs:
                    arg_mapping[arg_name] = actual_kwargs[arg_name]
                else:
                    arg_mapping[arg_name] = args_info['defaults'][i - (len(args_info['args']) - len(args_info['defaults']))]
        
        # 处理*args参数
        if args_info['varargs']:
            varargs_list = actual_args[len(args_info['args']):]
            if varargs_list:
                arg_mapping[args_info['varargs']] = f"({', '.join(varargs_list)})"
            else:
                arg_mapping[args_info['varargs']] = '()'
        
        # 处理仅关键字参数
        for i, arg_name in enumerate(args_info['kwonly_args']):
            if arg_name in actual_kwargs:
                arg_mapping[arg_name] = actual_kwargs[arg_name]
            else:
                arg_mapping[arg_name] = args_info['kwonly_defaults'][i]
        
        # 处理**kwargs参数
        if args_info['varkw']:
            used_kwargs = set(args_info['args']).union(args_info['kwonly_args'])
            remaining_kwargs = {k: v for k, v in actual_kwargs.items() if k not in used_kwargs}
            if remaining_kwargs:
                kwargs_items = [f"{k}: {v}" for k, v in remaining_kwargs.items()]
                arg_mapping[args_info['varkw']] = f"{{{', '.join(kwargs_items)}}}"
            else:
                arg_mapping[args_info['varkw']] = '{}'

        _log(f"Argument mapping: {arg_mapping}", level=DEBUG_LOG_LEVEL)
        
        # 修改方法体中的变量名
        modified_body = callee_method_body
        
        # 收集所有入参信息
        all_params = set(args_info['args'] + args_info['kwonly_args'])
        if args_info['varargs']:
            all_params.add(args_info['varargs'])
        if args_info['varkw']:
            all_params.add(args_info['varkw'])
        assigned_param_names = local_vars.intersection(all_params)
        assigned_param_aliases = {name: name for name in assigned_param_names}
        alias_to_param = {alias: param for param, alias in assigned_param_aliases.items()}
        
        # 1. 先处理入参的重命名
        param_name_mapping = {}
        other_param_mapping = {}
        
        # 处理位置参数和关键字参数
        for arg_name, arg_value in arg_mapping.items():
            if arg_name in assigned_param_names:
                continue
            # 如果实参是一个变量名，而不是常量或表达式
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', arg_value) and not arg_value.startswith('_'):
                param_name_mapping[arg_name] = arg_value
            else:
                ## self这种替换要在常量替换之后进行
                other_param_mapping[arg_name] = arg_value


        # 只重命名不是入参的局部变量
        local_only_vars = (local_vars - all_params) | assigned_param_names

        # 找出与caller中局部变量冲突的变量
        conflicting_vars = local_only_vars.intersection(caller_locals)

        # 处理局部变量
        for var in local_only_vars:
            if var in conflicting_vars:
                # 只有发生冲突时才重命名
                prefix = '_'
                new_name = f"{prefix}{var}"
                while new_name in caller_locals or new_name in local_vars or new_name in all_params:
                    prefix += '_'
                    new_name = f"{prefix}{var}"

                modified_body = self._safe_replace_identifier(
                    modified_body,
                    var,
                    new_name
                )
                if var in alias_to_param:
                    original_param = alias_to_param.pop(var)
                    assigned_param_aliases[original_param] = new_name
                    alias_to_param[new_name] = original_param
                local_vars.discard(var)
                local_vars.add(new_name)

        # replace the old name into new one in the callee body
        for old_name, new_name in param_name_mapping.items():
            modified_body = self._safe_replace_identifier(modified_body, old_name, new_name)

        for old_name, new_name in other_param_mapping.items():
            modified_body = self._safe_replace_identifier(modified_body, old_name, new_name)
        
        # add special variable initialization, for the method params will be reassigned in the callee body
        if assigned_param_names:
            ordered_assigned_params = []
            for name in args_info['args']:
                if name in assigned_param_names:
                    ordered_assigned_params.append(name)
            for name in args_info['kwonly_args']:
                if name in assigned_param_names:
                    ordered_assigned_params.append(name)
            if args_info['varargs'] and args_info['varargs'] in assigned_param_names:
                ordered_assigned_params.append(args_info['varargs'])
            if args_info['varkw'] and args_info['varkw'] in assigned_param_names:
                ordered_assigned_params.append(args_info['varkw'])

            initializer_lines = []
            for param in ordered_assigned_params:
                alias_name = assigned_param_aliases.get(param, param)
                initializer_value = arg_mapping.get(param, 'None')
                initializer_lines.append(f"{alias_name} = {initializer_value}")

            initializer_block = "\n".join(initializer_lines)
            if modified_body.strip():
                modified_body = initializer_block + "\n" + modified_body
            else:
                modified_body = initializer_block

        existing_names = set(all_params)
        existing_names.update(local_vars)
        existing_names.update(caller_locals)
        if return_var and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', return_var):
            existing_names.add(return_var)
        modified_body = self._rewrite_returns_with_flag(modified_body, return_var, existing_names)
        
        # 缩进内联的代码
        indent = len(original_call[0]) - len(original_call[0].lstrip())
        indented_body = '\n'.join([' ' * indent + line for line in modified_body.splitlines()])
        
        _log(f"\nReplaced with:", level=DEBUG_LOG_LEVEL)
        _log(indented_body, level=DEBUG_LOG_LEVEL)
        
        # 收集caller中的替换信息
        return {
            'start': start_line,
            'end': end_line,
            'replacement': indented_body.splitlines(),
            'imports': imports
        }
        
    def collect(self, all_calls, all_class_parents, family_classes):
        """
        @param callee_mapping: {(called_module, called_class, called_method_name): [(module_path, class_name, call_locations)]}
        @param all_calls: {(caller_method): {(called_method): [(module_path, class_name, call_locations)]}}
        
        @return: List of methods [{"type": xxx , "called": (module_path, class_name, method_name), "caller": call_locations}]
        """
        raise Exception("Not implemented")
