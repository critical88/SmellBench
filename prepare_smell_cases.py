"""
Prepare Smell Cases — One-click pipeline for generating smell benchmark cases.

Follows the same pattern as prepare_project.py:
1. Ensure base + per-project Docker images are built
2. Install the specified code agent inside the container
3. Run smell_benchmark.py inside each project's container
4. Collect results

Usage:
    python prepare_smell_cases.py --agent claude_code
    python prepare_smell_cases.py --agent codex --project-name click
    python prepare_smell_cases.py --agent openhands --force --max-workers 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path

import docker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open("repo_list.json") as f:
    repo_dict = json.load(f)

print_lock = threading.Lock()
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

client = docker.from_env()

BASE_IMAGE_TAG = "critical88/smellbench_base:latest"

SUPPORTED_AGENTS = ("claude_code", "qwen_code", "openhands", "codex")

# Environment variables each agent may need, forwarded into the container
AGENT_ENV_KEYS = {
    "claude_code": [
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    ],
    "qwen_code": [
        "QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_CODE_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
    ],
    "openhands": [
        "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
    ],
    "codex": [
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_MODEL",
    ],
}

# Common env keys forwarded for all agents (used by call_llm for analysis)
COMMON_ENV_KEYS = [
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
]


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def ensure_base_image() -> None:
    """Build the base image if it does not exist."""
    images = client.images.list()
    local_tag = "smellbench_base:latest"
    if any(local_tag in tag for img in images for tag in img.tags):
        print(f"Base image '{local_tag}' already exists.")
        return
    print("Building base image ...")
    image, _ = client.images.build(path=".", dockerfile="Dockerfile", tag=local_tag, rm=True)
    print(f"Built base image: {image.tags[0]}")


def ensure_project_image(project_name: str) -> str:
    """Build the per-project Docker image if needed. Returns the image tag."""
    project_tag = f"smellbench_{project_name}:latest"
    if any(project_tag in tag for img in client.images.list() for tag in img.tags):
        with print_lock:
            print(f"Image '{project_tag}' already exists.")
        return project_tag

    repo_info = repo_dict[project_name]
    repo_url = repo_info["url"]
    commit_id = repo_info["commit_id"]
    conda_env_create = repo_info.get("conda_env_create", "")
    env_name = repo_info.get("env_name", project_name)
    build_cmd = repo_info.get("build_cmd", "pip install -e .")

    if isinstance(conda_env_create, list):
        conda_env_create = " && ".join(conda_env_create)
    if isinstance(build_cmd, str):
        build_cmd = [build_cmd]
    build_cmd = " && ".join(f"conda run -n {env_name} {cmd}" for cmd in build_cmd)

    image_dir = Path("docker_images") / project_name
    os.makedirs(image_dir, exist_ok=True)

    dockerfile = f"""\
FROM {BASE_IMAGE_TAG}
WORKDIR /workspace/project

ARG REPO_URL={repo_url}
ARG COMMIT_ID={commit_id}

RUN git clone --recursive "$REPO_URL" {project_name} \\
&& cd {project_name} \\
&& git checkout "$COMMIT_ID"

WORKDIR /workspace/project/{project_name}
RUN {conda_env_create}
RUN {build_cmd}

RUN git config --global user.email "smellbench@example.com"
RUN git config --global user.name "smellbench"

CMD ["/bin/bash"]
"""
    with open(image_dir / "Dockerfile", "w") as f:
        f.write(dockerfile.strip() + "\n")

    with print_lock:
        print(f"Building image for project: {project_name}")
    image, _ = client.images.build(path=str(image_dir), tag=project_tag, rm=True)
    with print_lock:
        print(f"Built project image: {image.tags[0]}")
    return project_tag


# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------

def _collect_env_vars(agent: str) -> dict[str, str]:
    """Collect environment variables needed by the agent + common keys."""
    keys = set(COMMON_ENV_KEYS)
    keys.update(AGENT_ENV_KEYS.get(agent, []))
    env_vars = {}
    for key in keys:
        val = os.environ.get(key)
        if val:
            env_vars[key] = val
    return env_vars


def run_smell_benchmark_in_container(
    project_name: str,
    project_tag: str,
    agent: str = "claude_code",
    force: bool = False,
    model: str = "",
    base_url: str = "",
) -> tuple[str, bool]:
    """Run the smell benchmark pipeline inside a Docker container.

    Returns (project_name, success).
    """
    log_file_path = os.path.join(LOG_DIR, f"{project_name}_smell_{agent}.log")
    current_dir = os.getcwd()

    # Build the shell command
    cmd_parts = [f"bash scripts/run_smell_benchmark.sh {project_name}"]
    cmd_parts.append(f"--agent {agent}")
    if force:
        cmd_parts.append("--force")
    if model:
        cmd_parts.append(f"--model {model}")
    if base_url:
        cmd_parts.append(f"--base-url {base_url}")
    command = " ".join(cmd_parts)

    env_vars = _collect_env_vars(agent)

    with print_lock:
        print(f"Starting container for {project_name} (agent={agent})")

    container = client.containers.create(
        image=project_tag,
        command=command,
        working_dir="/workspace/smell",
        environment=env_vars,
        volumes={
            current_dir: {
                "bind": "/workspace/smell",
                "mode": "rw",
            }
        },
        tty=True,
    )

    try:
        container.start()

        with open(log_file_path, "wb") as f:
            for line in container.logs(stream=True):
                f.write(line)
                f.flush()

        result = container.wait()
        exit_code = result["StatusCode"]
        return project_name, exit_code == 0

    except Exception as e:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\nERROR: {e}\n")
        return project_name, False

    finally:
        container.remove(force=True)
        with print_lock:
            print(f"Container {project_name} removed.")


# ---------------------------------------------------------------------------
# One-click orchestrator
# ---------------------------------------------------------------------------

def process_one_project(
    project_name: str,
    agent: str = "claude_code",
    force: bool = False,
    model: str = "",
    base_url: str = "",
) -> tuple[str, bool]:
    """End-to-end: ensure image -> run benchmark -> return result."""
    try:
        project_tag = ensure_project_image(project_name)
        return run_smell_benchmark_in_container(
            project_name, project_tag,
            agent=agent, force=force, model=model, base_url=base_url,
        )
    except Exception as e:
        with print_lock:
            print(f"[ERROR] {project_name}: {e}")
        return project_name, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click smell benchmark case generation via Docker.",
    )
    parser.add_argument("--agent", default="claude_code",
                        choices=SUPPORTED_AGENTS,
                        help="Code agent to use for smell injection "
                             f"(choices: {', '.join(SUPPORTED_AGENTS)}).")
    parser.add_argument("--project-name", default=None,
                        help="Process a single repo instead of all selected.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists.")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Max parallel containers.")
    parser.add_argument("--model", default="",
                        help="Model for LLM calls "
                             "(e.g. anthropic/claude-sonnet-4-5-20250929).")
    parser.add_argument("--base-url", default="",
                        help="Base URL for LLM API.")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Docker image build, assume images exist.")
    args = parser.parse_args()

    # 1. Ensure base image
    if not args.skip_build:
        ensure_base_image()

    # 2. Select repos
    projects = []
    for name, spec in repo_dict.items():
        if args.project_name and name != args.project_name:
            continue
        if not args.project_name and not spec.get("selected", False):
            continue
        projects.append(name)

    if not projects:
        print("No repos selected. Check repo_list.json or --project-name.")
        return

    print(f"\nAgent:    {args.agent}")
    print(f"Projects: {', '.join(projects)} ({len(projects)} total)")
    print(f"Workers:  {args.max_workers}")
    print()

    # 3. Run smell benchmark for each project in parallel
    failed_projects = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_one_project,
                p,
                agent=args.agent,
                force=args.force,
                model=args.model,
                base_url=args.base_url,
            ): p
            for p in projects
        }
        for future in concurrent.futures.as_completed(futures):
            project_name, success = future.result()
            with print_lock:
                if success:
                    print(f"[SUCCESS] {project_name}")
                else:
                    print(f"[FAILED]  {project_name}")
                    failed_projects.append(project_name)

    # 4. Summary
    print(f"\n{'='*60}")
    print(f"Smell Case Generation Summary (agent={args.agent})")
    print(f"{'='*60}")
    print(f"  Total:   {len(projects)}")
    print(f"  Success: {len(projects) - len(failed_projects)}")
    print(f"  Failed:  {len(failed_projects)}")

    if failed_projects:
        print("\nFailed projects:")
        for p in failed_projects:
            log_path = os.path.join(LOG_DIR, f"{p}_smell_{args.agent}.log")
            print(f"  - {p}  (log: {log_path})")
    else:
        print("\nAll projects succeeded!")


if __name__ == "__main__":
    main()
