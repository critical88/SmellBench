
# SmellBench: Can Code Agents Smell and Refactor Bad Code?

This repository provides the **official PyTorch implementation** of **SmellBench**, a benchmark designed to evaluate whether code agents can detect and refactor bad code (code smells).

## Dataset Format

Each instance in **SmellBench** represents a validated code smell injection case constructed from a real-world open-source repository.

Each sample contains:

* Repository metadata
* Injected code smell information
* Related test cases
* Code before refactoring (with smell)
* Ground-truth refactored code (structured as a call tree)

Below is a complete example of a single data instance.

---

### Full JSON Structure (Single Sample)

```json
{
  "instance_id": "8a580458bac9d4bd6fc0ee8094786c36",
  "type": "Long",
  "smell_content": "diff --git a/src/xxx.py b/src/xxx.py ...",

  "project_name": "click",
  "commit_hash": "1d038f270701498433cb432f54db89f95f07a845",

  "meta": {
    "key": "depth_1",
    "depth": 1,
    "calling_times": 3
  },

  "testsuites": [
    "test_utils.py::test_make_default"
  ],

  "before_refactor_code": {
    "type": "caller",
    "start": 1096,
    "end": 1151,
    "code": "def get_short_help_str(...): ...",
    "module_path": "click.core",
    "class_name": "Command",
    "method_name": "get_short_help_str"
  },

  "after_refactor_code": [
    {
      "type": "caller",
      "code": "def get_short(...): ...",
      "position": {
        "module_path": "click.core",
        "class_name": "Command"
      },

      "callees": [
        {
          "type": "callee",
          "start": 1103,
          "end": 1104,
          "code": "def make_default(...): ...",

          "position": {
            "module_path": "click.core",
            "class_name": "Command"
          },

          "callees": []
        }
      ]
    }
  ]
}
```

---

### Field Description

#### Top-level Fields

* **`instance_id`**
  Unique identifier for the sample.

* **`type`**
  Code smell category. Currently supported:

  * `"Long"` (Long Method)
  * `"Duplicated"` (Duplicated Code)

* **`smell_content`**
  The injected patch (Git diff format) introducing the code smell.

* **`project_name`**
  Name of the source repository.

* **`commit_hash`**
  Specific commit used to ensure reproducibility.

---

#### `meta` Field

Metadata related to cascade injection:

* **`key`**
  Injection category: `depth_1`, `depth_2`, `depth_3`, or `duplicated`.

* **`depth`**
  Depth of the call chain in cascade injection.

* **`calling_times`**
  Number of callees invoked by the caller.

---

#### `testsuites`

A list of related unit tests.

All injected samples are validated to pass these tests to ensure behavioral equivalence after refactoring.

---

### Code Representation

#### `before_refactor_code`

Represents the **smell-injected function** (the target to be refactored).

It is a single flat function definition containing:

* Source code
* Line range in the original file
* Module path
* Class name
* Method name

---

#### `after_refactor_code` (Ground Truth)

Represents the **ideal refactored structure**.

Unlike the flat representation before refactoring, this is modeled as a **call tree**.

### Tree Structure

```
caller
 └── callee
      └── subcallee
           └── ...
```

Each node (caller or callee) contains:

* `type` (caller / callee)
* `code` (source code)
* `position` (module path + class name)
* `callees` (nested child functions)

This recursive structure enables representing:

* Single-level extraction (`depth_1`)
* Multi-level cascade extraction (`depth_2`, `depth_3`)
* Duplicated code extraction

---

### Before vs After Refactoring

| Before Refactoring      | After Refactoring                     |
| ----------------------- | ------------------------------------- |
| Long function or Duplicated Code    | Decomposed into multiple functions    |
| Flat representation     | Hierarchical call-tree representation |
| Contains injected smell | Clean, structured implementation      |

---




#  Benchmark Pipeline

## One-Click Reproducibility
To ensure full reproducibility and minimize environment-related inconsistencies, we provide a one-click script:
```bash
python -u prepare_project.py
```

This script automatically executes the entire pipeline for all predefined repositories, including:

- Repository cloning

- Environment setup

- AST-based analysis

- Smell injection

- Data validation

- Benchmark sampling

All steps are executed inside a Docker container, which guarantees:

✅ Reproducibility across different machines

✅ Isolation from local dependency conflicts

✅ Ease of use (no manual environment configuration required)

✅ Portability and deployment consistency

Users only need Docker installed to reproduce the complete benchmark construction process.

Below we provide a detailed explanation of each stage in the pipeline.

## Environment Setup    

We recommend using **Python 3.12**.

Install the required dependencies with:

```bash
pip install -r repo_requirements.txt
```

## Repository Selection

We provide metadata for selected repositories in `repo_list.json`.

If you would like to add additional repositories, please extend the configuration in this file. Below is an example:

```json
{
  "matplotlib":{ 
        "url": "https://github.com/matplotlib/matplotlib",  
        "commit_id": "1392cbe3c79cdb93f9282747841d648770f60249",
        "src_path": "lib/matplotlib",
        "conda_env_create": "conda env create --file environment.yml",
        "build_cmd": [
                        "python -m pip install --verbose --no-build-isolation --editable \".[dev]\""
                    ],
        "env_name": "mpl-dev"
    }
}
```

### Field Description

* `url`: Repository GitHub URL
* `commit_id`: Specific commit version used for reproducibility
* `src_path`: Source code directory for analysis
* `conda_env_create`: Command to create the repository environment
* `build_cmd`: Installation commands for the repository
* `env_name`: Conda environment name

---

## AST-based Analysis

Before generating data, we perform dependency and AST analysis on the repository.

Run:

```bash
python -u ast_analyze.py --project-name matplotlib
```

This command will automatically:

* Clone the repository
* Create and install the required environment
* Build and install the project

After analysis is completed, the parsed results will be saved to:

```
output/{project-name}/function_testunit_mapping.json
```

⚠️ This step must be executed **once per repository**.
Future updates will migrate this process into a Docker-based environment.

---

## Smell Injection

In this stage, we inject predefined code smells into the selected repositories.

Run:

```bash
python -u smell_injection.py --project-name matplotlib
```

This step will:

* Identify injectable code regions
* Inject predefined smells
* Generate candidate (original, modified) code pairs
* Validate each pair using `unittest`

Only data pairs that pass unit tests are retained in the candidate pool.

---

## Data Sampling

Since the collected dataset can be large, we apply a sampling strategy to construct the final benchmark.

Run:

```bash
python -u sample_data.py
```

The final benchmark file will be generated at:

```
outputs/benchmark.json
```

---

# Evaluation

After obtaining the benchmark, we provide an evaluation pipeline for testing code agents.

Run:

```bash
python -u evaluate.py \
    --model qwen_code \
    --llm_model gpt5 \
    --api_key xxx \
    --base_url xxx
```

This will generate the evaluation results for the specified code agent.

---

# Notes

* Each repository requires independent preprocessing.
* Make sure all repository environments are correctly created before running injection.
* API-based models require valid API credentials.

---



