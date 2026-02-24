from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict
from utils import list_conda_envs
import json
import os
import docker
with open("repo_list.json") as f:
    repo_list = json.load(f)

DEFAULT_PROJECT_DIR = Path("../project")
client = docker.from_env()

def build_base_image() -> None:
    image, logs = client.images.build(path=".", dockerfile="Dockerfile", tag="smellbench_base:latest")
    print(f"Built base image: {image.tags[0]}")
def build_repo_images() -> None:
    base_image_dir = Path("docker_images")
    print(f"Creating Dockerfile for projects")
    for project_name, item in repo_list.items():

        repo_info = item
        repo_url = repo_info.get("url")
        commitid = repo_info.get("commit_id")
        conda_env_create = repo_info.get("conda_env_create") # "conda create -n click-dev python==3.10 pytest pytest-cov"
        env_name = repo_info.get("env_name") # click-dev
        build_cmd = repo_info.get("build_cmd", "pip install -e .") #. "pip install -e \".[dev]\"",
        
        image_dir = base_image_dir / project_name
        os.makedirs(image_dir, exist_ok=True)

        # Dockerfile
        os.makedirs(image_dir, exist_ok=True)
        dockerfile = f"""\
FROM smellbench_base:latest
WORKDIR /workspace/project

ARG REPO_URL={repo_url}
ARG COMMIT_ID={commitid}

RUN git clone --recursive "$REPO_URL" {project_name} \
    && cd {project_name} \
    && git checkout "$COMMIT_ID"

WORKDIR /workspace/project/{project_name}
RUN {conda_env_create}
RUN conda run -n {env_name} {build_cmd}

CMD [\"/bin/bash\"]
"""
        with open(os.path.join(image_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile.strip() + "\n")
    
    print(f"Building images for projects")
    for project_name, item in repo_list.items():
        
        image_dir = base_image_dir / project_name

        image, logs = client.images.build(path=str(image_dir), tag=f"smellbench_{project_name}:latest")

        print(f"Built project image: {image.tags[0]}")

def main() -> None:
    
    build_base_image()

    build_repo_images()


if __name__ == "__main__":
    main()