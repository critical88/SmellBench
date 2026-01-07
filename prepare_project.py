from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict

repo_list: Dict[str, str] = {
    "click": {
        "url": "https://github.com/pallets/click.git",
        "commit_id": "1d038f270701498433cb432f54db89f95f07a845"
    },
    "numpy": {
        "url": "https://github.com/numpy/numpy.git",
        "commit_id": "c3d60fc8393f3ca3306b8ce8b6453d43737e3d90"
    },
    "requests": {
        "url": "https://github.com/psf/requests.git",
        "commit_id": "b25c87d7cb8d6a18a37fa12442b5f883f9e41741"
    },
    "urllib3": {
        "url": "https://github.com/urllib3/urllib3.git",
        "commit_id": "83f8643ffb5b7f197457379148e2fa118ab0fcdc"
    },
    "jinja": {
        "url": "https://github.com/pallets/jinja.git",
        "commit_id": "15206881c006c79667fe5154fe80c01c65410679"
    },
    "pydantic": {
        "url": "https://github.com/pydantic/pydantic.git",
        "commit_id": "d771df98f3194ea2695a6a893e318b32381032c3"
    }
}

DEFAULT_PROJECT_DIR = Path("../project")


def clone_repos(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)

    for name, item in repo_list.items():
        url = item['url']
        commit_id = item['commit_id']
        repo_path = project_dir / name
        if repo_path.exists():
            print(f"[skip] {name}: {repo_path} 已存在，跳过")
            continue

        print(f"[clone] {name}: {url} -> {repo_path}")
        try:
            subprocess.run(["git", "clone", url, str(repo_path)], check=True)
            subprocess.run(["git", "checkout", commit_id], cwd=name, check=True)
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
