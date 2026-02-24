
# SmellBench: Can Code Agents Smell and Refactor Bad Code?

This repository provides the **official PyTorch implementation** of **SmellBench**, a benchmark designed to evaluate whether code agents can detect and refactor bad code (code smells).

---

## 📦 Environment Setup

We recommend using **Python 3.12**.

Install the required dependencies with:

```bash
pip install -r repo_requirements.txt
```

---

# 🔬 Benchmark Pipeline

The benchmark construction consists of the following steps:

---

## 1️⃣ Repository Selection

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

## 2️⃣ AST-based Analysis

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

## 3️⃣ Smell Injection

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

## 4️⃣ Data Sampling

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

# 🚀 Evaluation

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

# 📌 Notes

* Each repository requires independent preprocessing.
* Make sure all repository environments are correctly created before running injection.
* API-based models require valid API credentials.

---



