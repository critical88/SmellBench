import random
import os
import json
import uuid
import hashlib

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
        "hard":10,
        "duplicated":2
    },
    "pandas":{
        "simple": 8,
        "medium": 12,
        "hard":15,
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
        "hard":15,
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
    # "sympy":{
    #     "simple": 8,
    #     "medium": 12,
    #     "hard":15,
    #     "duplicated":2
    # },
}
benchmark_file = "output/benchmark.jsonl"
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