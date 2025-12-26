from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict

repo_list: Dict[str, str] = {
    "click": "https://github.com/pallets/click.git",
    "numpy": "https://github.com/numpy/numpy.git",
    "requests": "https://github.com/psf/requests.git",
    "urllib3": "https://github.com/urllib3/urllib3.git",
    "jinja": "https://github.com/pallets/jinja.git",
}

DEFAULT_PROJECT_DIR = Path("../project")


def clone_repos(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)

    for name, url in repo_list.items():
        repo_path = project_dir / name
        if repo_path.exists():
            print(f"[skip] {name}: {repo_path} 已存在，跳过")
            continue

        print(f"[clone] {name}: {url} -> {repo_path}")
        try:
            subprocess.run(["git", "clone", url, str(repo_path)], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[error] {name} 克隆失败: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="克隆 repo_list 中列出的仓库到 project 目录（默认在项目根目录下）"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="自定义克隆目标目录（默认使用仓库根目录下的 project）",
    )
    args = parser.parse_args()
    clone_repos(args.dest.expanduser().resolve())


if __name__ == "__main__":
    main()
