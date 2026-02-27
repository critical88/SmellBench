from __future__ import annotations

from pathlib import Path
from typing import Dict
from utils import list_conda_envs
import json
import os
import docker
import concurrent.futures
import threading
with open("repo_list.json") as f:
    repo_dict = json.load(f)

print_lock = threading.Lock()
DEFAULT_PROJECT_DIR = Path("../project")
client = docker.from_env()
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def build_base_image() -> None:
    print(f"Building base image")
    image, logs = client.images.build(path=".", dockerfile="Dockerfile", tag="smellbench_base:latest")
    print(f"Successfully built base image: {image.tags[0]}")

def build_one_repo(project_name):
    
    base_image_dir = Path("docker_images")
    project_tag = f"smellbench_{project_name}:latest"
    repo_info = repo_dict[project_name]
    repo_url = repo_info.get("url")
    commitid = repo_info.get("commit_id")
    conda_env_create = repo_info.get("conda_env_create") # "conda create -n click-dev python==3.10 pytest pytest-cov"
    env_name = repo_info.get("env_name") # click-dev
    build_cmd = repo_info.get("build_cmd", "pip install -e .") #. "pip install -e \".[dev]\"",
    
    image_dir = base_image_dir / project_name
    os.makedirs(image_dir, exist_ok=True)
    if isinstance(conda_env_create, list):
        conda_env_create = " && ".join(conda_env_create)
    if isinstance(build_cmd, list):
        build_cmd = [ f"conda run -n {env_name} {cmd}" for cmd in build_cmd ]
        build_cmd = " && ".join(build_cmd)
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
RUN {build_cmd}

CMD [\"/bin/bash\"]
"""
    with open(os.path.join(image_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile.strip() + "\n")
    
    print("Building image for project:", project_name)

    image_dir = base_image_dir / project_name

    image, logs = client.images.build(path=str(image_dir), tag=project_tag)

    print(f"Successfully built project image: {image.tags[0]}")

    run_construct(project_name, project_tag)

def build_repo_images() -> None:
    
    print(f"Creating Dockerfile for projects")
    projects = list(repo_dict.keys())
    # projects = ["click", "jinja", "numpy"]

    max_workers = 3

    failed_projects = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_one_repo, p): p
            for p in projects
        }
        for future in concurrent.futures.as_completed(futures):
            project, success = future.result()

            with print_lock:
                if success:
                    print(f"[SUCCESS] {project}")
                else:
                    print(f"[FAILED]  {project}")
                    failed_projects.append(project)
    if len(failed_projects) == 0:
        print("\nAll projects succeeded. Running next step...")
        run_next_step()
    else:
        print("\nSome projects failed:")
        for p in failed_projects:
            print(" -", p)

def run_next_step():
    print(">>> Executing next pipeline step...")
    from sample_data import sample_data
    sample_data()
def run_construct(project_name, project_tag):
    log_file_path = os.path.join(LOG_DIR, f"{project_name}.log")
    client = docker.from_env()
    current_dir = os.getcwd()

    print(f"Starting container for project: {project_name}")

    container = client.containers.create(
        image=project_tag,  # 你的镜像名
        command=f"bash scripts/run_construct.sh {project_name}",
        working_dir="/workspace/smell",
        volumes={
            current_dir: {
                "bind": "/workspace/smell",
                "mode": "rw"
            }
        },
        tty=True
    )

    try:
        container.start()

        with open(log_file_path, "w", encoding="utf-8") as f:
            for line in container.logs(stream=True):
                decoded = line.decode()
                f.write(decoded)
                f.flush()

        result = container.wait()
        exit_code = result["StatusCode"]

        # container.remove()

        if exit_code == 0:
            return project_name, True
        else:
            return project_name, False

        return exit_code
    except Exception as e:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\nERROR: {str(e)}\n")
        
        return project_name, False
    finally:
        container.remove(force=True)
        print(f"Container {project_name} removed.")
def main() -> None:
    
    build_base_image()

    build_repo_images()



if __name__ == "__main__":
    main()