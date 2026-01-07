import contextlib
from pathlib import Path
import os
import textwrap
import ast

@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def strip_python_comments(text: str) -> str:
    stripped = textwrap.dedent(text)
    if not stripped.strip():
        return stripped
    try:
        tree = ast.parse(stripped)
        method_ast = tree.body[0]
        body_without_docstring = [node for node in method_ast.body if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Str)]
        return ast.unparse(body_without_docstring)
        
    except Exception:
        return text