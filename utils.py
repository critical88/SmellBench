import contextlib
from pathlib import Path
import os
import textwrap
import ast
import shutil
import logging
import uuid
import hashlib

INFO_LOG_LEVEL=logging.INFO
DEBUG_LOG_LEVEL=logging.DEBUG

@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change directories."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

@contextlib.contextmanager
def disableGitTools(project_path: Path):
    git_dir = project_path / ".git"
    tmp_dir = project_path.parent / f"{project_path.name}_{str(uuid.uuid4())}"
    dest_dir = tmp_dir / ".git"
    os.makedirs(tmp_dir, exist_ok=True)
    move = False
    if git_dir.exists():
        shutil.move(git_dir, dest_dir)
        move = True
    try:
        yield
    finally:
        if dest_dir.exists() and move:
            shutil.move(dest_dir, git_dir)
        os.removedirs(tmp_dir)

def _log(text, level=INFO_LOG_LEVEL):
    logging.log(level, text)
    


def hashcode(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


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