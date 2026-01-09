from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict
import json
with open("repo_list.json") as f:
    repo_list = json.load(f)

DEFAULT_PROJECT_DIR = Path("../project")


def clone_repos(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)

    for name, item in repo_list.items():
        url = item['url']
        commit_id = item['commit_id']
        repo_path = project_dir / name
        if repo_path.exists():
            print(f"[skip] {name}: {repo_path} already exists")
        else:
            print(f"[clone] {name}: {url} -> {repo_path}")
            try:
                subprocess.run(["git", "clone", url, str(repo_path)], check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[error] {name} failed to clone: {exc}")
        
        try:
            subprocess.run(["git", "checkout", commit_id], cwd=str(repo_path.absolute()), check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[error] checkout failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clone repo from repo list into project dir"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="default is `project`, the path that git clone into",
    )
    args = parser.parse_args()
    clone_repos(args.dest.expanduser().resolve())


if __name__ == "__main__":
    main()