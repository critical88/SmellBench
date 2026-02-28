import random
import os
import json
import uuid
import hashlib
import shutil

from eval_utils import build_prompt
SEED = 42
random.seed(42)

sample_rules = {
    "click":{
        "simple": 8,
        "medium": 10,
        "hard":4,
        "duplicated":1
    },
    "jinja":{
        "simple": 8,
        "medium": 4,
        "hard":0,
        "duplicated":2
    },
    "seaborn":{
        "simple": 8,
        "medium": 10,
        "hard":8,
        "duplicated":2
    },
    "pandas":{
        "simple": 8,
        "medium": 12,
        "hard":16,
        "duplicated":2
    },
    "matplotlib":{
        "simple": 8,
        "medium": 10,
        "hard":10,
        "duplicated":2
    },
    "sphinx":{
        "simple": 8,
        "medium": 10,
        "hard":10,
        "duplicated":2
    },
    "scikit-learn":{
        "simple": 8,
        "medium": 12,
        "hard":16,
        "duplicated":3
    },
    "numpy":{
        "simple": 8,
        "medium": 10,
        "hard":10,
        "duplicated":2
    },
    "xarray":{
        "simple": 8,
        "medium": 10,
        "hard":10,
        "duplicated":2
    },
    "sympy":{
        "simple": 8,
        "medium": 12,
        "hard":16,
        "duplicated":2
    },
}

def sample_data():
    benchmark_file = os.path.join("output", "benchmark.jsonl")
    benchmark = []
    for repo_name, rules in sample_rules.items():
        repo_file = os.path.join("output",repo_name,  "successful_refactor_codes.json")
        if not os.path.exists(repo_file):
            print(f"{repo_name} didn't prepare the data")
            continue
        with open(repo_file) as f:
            repo_data = json.load(f)
        simple_data = []
        medium_data = []
        hard_data = []
        duplicated_data = []
        for d in repo_data["refactor_codes"]:
            d['name'] = repo_name
            d['settings'] = repo_data['settings']
            case_id = f"{d['name']}-{d['type']}-" + (hashlib.md5(json.dumps(d).encode("utf-8")).hexdigest())
            d['instance_id'] = case_id
            if d['type'] == "Long":
                if d['meta']['depth'] == 1:
                    simple_data.append(d)
                elif d['meta']['depth'] == 2:
                    medium_data.append(d)
                elif d['meta']['depth'] == 3:
                    hard_data.append(d)
            elif d['type'] == 'Duplicated':
                duplicated_data.append(d)
        
        if len(simple_data) > rules['simple']:
            simple_data = random.sample(simple_data, k=rules['simple'])
        if len(medium_data) > rules['medium']:
            medium_data = random.sample(medium_data, k=rules['medium'])
        if len(hard_data) > rules['hard']:
            hard_data = random.sample(hard_data, k=rules['hard'])
        if len(duplicated_data) > rules['duplicated']:
            duplicated_data = random.sample(duplicated_data, k=rules['duplicated'])
        benchmark.extend(simple_data)
        benchmark.extend(medium_data)
        benchmark.extend(hard_data)
        benchmark.extend(duplicated_data)

        print(f"{repo_name} has {len(simple_data) + len(medium_data) + len(hard_data) + len(duplicated_data)} data")

    with open(benchmark_file, "w") as f:
        for b in benchmark:
            f.write(json.dumps(b) + "\n")
    
    return benchmark_file

def adapter(benchmark_file: str):
    benchmark = []
    with open(benchmark_file, "r") as f:
        for line in f.readlines():
            benchmark.append(json.loads(line.strip()))
    # ==============================================================
    # After sampling: generate structured dataset (SWE-bench style)
    # ==============================================================
    import toml
    ROOT = "./"
    HARBOR_DIR = os.path.join(ROOT, "harbor")
    REPO_LIST_PATH = os.path.join(ROOT, "repo_list.json")
    TEMPLATE_DIR = os.path.join(HARBOR_DIR, "templates")
    DATASET_DIR = os.path.join(HARBOR_DIR, "SmellBench")
    os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.exists(REPO_LIST_PATH):
        print(f"[!] repo_list.json not found at {REPO_LIST_PATH}")
        return
    
    with open(REPO_LIST_PATH, "r") as f:
        repo_dict = json.load(f)

    docker_template_path = os.path.join(TEMPLATE_DIR, "Dockerfile")
    register_tasks = []
    for b in benchmark:
        use_code_agent = True
        use_test = False
        instance_id = b['instance_id']
        repo_name = b['name']
        repo_info = repo_dict[repo_name]
        repo_url = repo_info.get("url")
        smell_type = repo_info.get("type", "unknown")
        commitid = repo_info.get("commit_id")
        conda_env_create = repo_info.get("conda_env_create") # "conda create -n click-dev python==3.10 pytest pytest-cov"
        env_name = repo_info.get("env_name") # click-dev
        build_cmd = repo_info.get("build_cmd", "pip install -e .") #. "pip install -e \".[dev]\"",
        task_name = f"task-{instance_id}".lower()
        task_dir = os.path.join(DATASET_DIR, task_name)
        os.makedirs(task_dir, exist_ok=True)


        prompt = build_prompt(b,  use_code_agent=use_code_agent, use_test=use_test)

        # 1. instruction.md
        instruction = prompt
        with open(os.path.join(task_dir, "instruction.md"), "w") as f:
            f.write(instruction)

        # 2. task.toml
        task_meta = {
            "version": "0.1",
            "metadata":{
                "repo_url": repo_url,
                "commit_id": commitid,
                "language": repo_info.get("language", "python"),
                "build_env": conda_env_create,
                "build_repo": build_cmd,
            },
            "verifier":{
                "timeout_sec": 30
            },
            "agent":{
                "timeout_sec": 600
            },
            "environment":{
                "build_timeout_sec": 600,
                "docker_image": f"smellbench_{repo_name}:latest",
                "allow_internet": True

            }
        }
        with open(os.path.join(task_dir, "task.toml"), "w") as f:
            toml.dump(task_meta, f)
        
        register_tasks.append({
            "name": task_name,
            "git_url": repo_url,
            "git_commit_id": commitid,
            "path": task_name
        })

        # 3. environment/Dockerfile
        env_dir = os.path.join(task_dir, "environment")
        os.makedirs(env_dir, exist_ok=True)
        shutil.copy(f"docker_images/{repo_name}/Dockerfile", os.path.join(env_dir, "Dockerfile"))
        # with open(os.path.join(env_dir, "Dockerfile"), "w") as f:
        #     f.write(dockerfile.strip() + "\n")

        # 4. solution/solve.sh
        sol_dir = os.path.join(task_dir, "solution")
        os.makedirs(sol_dir, exist_ok=True)
        with open(os.path.join(sol_dir, "solve.sh"), "w") as f:
            f.write("#!/bin/bash\n# placeholder for solver script\n")

        # 5. tests/test.sh
        test_dir = os.path.join(task_dir, "tests")
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, "test.sh"), "w") as f:
            f.write("#!/bin/bash\n# placeholder for tests\n")

        break

    print(f"[✓] Structured dataset generated at: {DATASET_DIR}")
    register_info = {
        "name": "SmellBench",
        "version": "0.1",
        "description": "A benchmark of code smell refactoring tasks across 10 popular Python repositories, covering various types of code smells and difficulty levels.",
        "task_count": len(benchmark),
        "tasks": register_tasks
    }
    with open(os.path.join(DATASET_DIR, "register.json"), "w") as f:
        json.dump(register_info, f, indent=4)

if __name__ == "__main__":
    benchmark_file = sample_data()
    # adapter(benchmark_file)
