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
import docker.errors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open("repo_list.json") as f:
    repo_dict = json.load(f)

print_lock = threading.Lock()
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

client = docker.from_env()

# Language-specific base image tags
BASE_IMAGE_TAGS = {
    "python": "critical88/smellbench_base_python:latest",
    "java": "critical88/smellbench_base_java:latest",
    "go": "critical88/smellbench_base_go:latest",
}

SUPPORTED_AGENTS = ("claude_code", "qwen_code", "openhands", "codex", "mock", "test")

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

def ensure_base_image(language: str = "python") -> str:
    """Build the language-specific base image if it does not exist.

    Args:
        language: Programming language (python, java, go)

    Returns:
        The image tag for the base image
    """
    language = language.lower()
    local_tag = BASE_IMAGE_TAGS.get(language, BASE_IMAGE_TAGS["python"])

    images = client.images.list()
    if any(local_tag in tag for img in images for tag in img.tags):
        print(f"Base image '{local_tag}' already exists.")
        return local_tag

    # Select Dockerfile based on language
    if language == "python":
        dockerfile = "Dockerfile"
    elif language == "java":
        dockerfile = "Dockerfile.java"
    elif language == "go":
        dockerfile = "Dockerfile.go"
    else:
        raise ValueError(f"Unsupported language: {language}")

    print(f"Building base image '{local_tag}' from {dockerfile} ...")
    image, _ = client.images.build(path=".", dockerfile=dockerfile, tag=local_tag, rm=True)
    print(f"Built base image: {image.tags[0]}")
    return local_tag


def ensure_project_image(project_name: str) -> str:
    """Ensure the per-project Docker image exists. Returns the image tag.

    Priority:
    1. Try to pull from Docker Hub: critical88/smellbench_{project_name}:latest
    2. Check for local image: smellbench_{project_name}:latest
    3. Build from scratch if neither exists
    """
    project_tag = f"smellbench_{project_name}:latest"
    remote_tag = f"critical88/smellbench_{project_name}:latest"

    # Step 1: Try to pull from Docker Hub
    try:
        with print_lock:
            print(f"Attempting to pull '{remote_tag}' from Docker Hub...")
        image = client.images.pull(remote_tag)
        # Tag it locally for convenience
        image.tag(project_tag.split(':')[0], tag=project_tag.split(':')[1])
        with print_lock:
            print(f"Successfully pulled and tagged: {remote_tag} -> {project_tag}")
        return project_tag
    except docker.errors.NotFound:
        with print_lock:
            print(f"Image '{remote_tag}' not found on Docker Hub.")
    except docker.errors.APIError as e:
        with print_lock:
            print(f"Failed to pull '{remote_tag}': {e}")

    # Step 2: Check for local image
    if any(project_tag in tag for img in client.images.list() for tag in img.tags):
        with print_lock:
            print(f"Using existing local image: '{project_tag}'")
        return project_tag

    # Step 3: Build from scratch
    with print_lock:
        print(f"No remote or local image found. Building '{project_tag}' from scratch...")

    repo_info = repo_dict[project_name]
    repo_url = repo_info["url"]
    commit_id = repo_info["commit_id"]
    language = repo_info.get("language", "python").lower()

    # Get the appropriate base image for this language
    base_image = BASE_IMAGE_TAGS.get(language, BASE_IMAGE_TAGS["python"])

    # Handle different project types (Python, Java, Go, etc.)
    if language == "python":
        conda_env_create = repo_info.get("conda_env_create", "")
        env_name = repo_info.get("env_name", project_name)
        build_cmd = repo_info.get("build_cmd", "pip install -e .")

        if isinstance(conda_env_create, list):
            conda_env_create = " && ".join(conda_env_create)
        if isinstance(build_cmd, str):
            build_cmd = [build_cmd]
        build_cmd = " && ".join(f"conda run -n {env_name} {cmd}" for cmd in build_cmd)

        dockerfile = f"""\
FROM {base_image}
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
    elif language == "java":
        build_cmd = repo_info.get("build_cmd", "mvn compile -DskipTests")
        jacoco_in_pom = repo_info.get("jacoco_in_pom", False)

        # Download JaCoCo agent if the project doesn't have it configured in pom.xml
        # This is needed for projects without JaCoCo in pom.xml that use JAVA_TOOL_OPTIONS injection
        jacoco_download = ""
        if not jacoco_in_pom:
            jacoco_download = "# Download JaCoCo agent for coverage collection (project doesn't have JaCoCo in pom.xml)\nRUN mvn dependency:get -Dartifact=org.jacoco:org.jacoco.agent:0.8.12:jar:runtime || true\n\n"

        dockerfile = f"""\
FROM {base_image}
WORKDIR /workspace/project

ARG REPO_URL={repo_url}
ARG COMMIT_ID={commit_id}

RUN git clone --recursive "$REPO_URL" {project_name} \\
&& cd {project_name} \\
&& git checkout "$COMMIT_ID"

WORKDIR /workspace/project/{project_name}

# Build the project
RUN {build_cmd}

{jacoco_download}RUN git config --global user.email "smellbench@example.com"
RUN git config --global user.name "smellbench"

CMD ["/bin/bash"]
"""
    elif language == "go":
        build_cmd = repo_info.get("build_cmd", "go build ./...")

        dockerfile = f"""\
FROM {base_image}
WORKDIR /workspace/project

ARG REPO_URL={repo_url}
ARG COMMIT_ID={commit_id}

RUN git clone --recursive "$REPO_URL" {project_name} \\
&& cd {project_name} \\
&& git checkout "$COMMIT_ID"

WORKDIR /workspace/project/{project_name}

# Download dependencies
RUN go mod download || true

# Build the project
RUN {build_cmd} || true

RUN git config --global user.email "smellbench@example.com"
RUN git config --global user.name "smellbench"

CMD ["/bin/bash"]
"""
    else:
        raise ValueError(f"Unsupported language: {language}")

    image_dir = Path("docker_images") / project_name
    os.makedirs(image_dir, exist_ok=True)

    with open(image_dir / "Dockerfile", "w") as f:
        f.write(dockerfile.strip() + "\n")
    with print_lock:
        print(f"Building image for project: {project_name} (language: {language})")
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

    # 1. Ensure base images for all languages used by selected repos
    if not args.skip_build:
        # Collect all languages used by selected repos
        languages_needed = set()
        for name, spec in repo_dict.items():
            if args.project_name and name != args.project_name:
                continue
            if not args.project_name and not spec.get("selected", False):
                continue
            language = spec.get("language", "python").lower()
            languages_needed.add(language)

        print(f"Building base images for languages: {', '.join(sorted(languages_needed))}")
        for lang in sorted(languages_needed):
            ensure_base_image(lang)

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
