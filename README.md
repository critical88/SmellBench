# SmellBench: Towards Fine-Grained Evaluation of Code Agents on Refactoring Tasks

This repository provides the benchmark for **SmellBench**, designed to evaluate whether code agents can detect and refactor bad code (code smells).

## Dataset Format

Each instance in **SmellBench** represents a validated code smell injection case constructed from a real-world open-source repository.

Each sample contains:

* Repository metadata
* Code smell type and difficulty
* Injected code smell information
* Target function and test cases
* Ground-truth refactored code (as a reversal diff)
* Detailed smell analysis

Below is a complete example of a single data instance.

---

### Full JSON Structure (Single Sample)

```json
{
  "instance_id": "click-feature_envy-abbada6d83f399a175bfbf64b8a402e5",
  "type": "feature_envy",
  "difficulty": "hard",
  "target_file": "src/click/core.py",
  "hint_targeted": "The `finalize_context` method in the `_ParseResultAdapter` class (src/click/parser.py) exhibits feature envy - please address this code smell.",
  "hint_guided": "Can you resolve the feature envy code smell present in src/click/parser.py?",
  "smell_function": [
    "src/click/parser.py",
    "_ParseResultAdapter",
    "finalize_context"
  ],
  "test_functions": [
    ["src/click/parser.py", "_ParseResultAdapter", "finalize_context"]
  ],
  "testsuites": [
    "tests/test_shell_completion.py::test_full_complete[...]"
  ],
  "smell_content": "diff --git a/src/click/_utils.py b/src/click/_utils.py\n...",
  "gt_content": "diff --git a/src/click/_utils.py b/src/click/_utils.py\n...",
  "hash": "abbada6d83f399a175bfbf64b8a402e5",
  "commit_hash": "1d038f270701498433cb432f54db89f95f07a845",
  "project_name": "click",
  "settings": {
    "src_path": "src/click",
    "commit_id": "1d038f270701498433cb432f54db89f95f07a845",
    "test_cmd": "",
    "envs": {
      "PYTHONPATH": "src"
    },
    "env_name": "click-dev"
  },
  "smell_analysis": "## Individual Change Analysis\n..."
}
```

---

### Field Description

#### Top-level Fields

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | string | Unique identifier (format: `{project}-{type}-{hash}`) |
| `type` | string | Code smell category (see supported types below) |
| `difficulty` | string | Difficulty level: `easy`, `medium`, or `hard` |
| `hint_targeted` | string | Targeted hint identifying the specific smell location |
| `hint_guided` | string | Guided hint for refactoring without specific location |
| `smell_function` | list | Location of smelly code: `[file_path, class_name, method_name]` |
| `test_functions` | list | Related test functions as `[file, class, method]` tuples |
| `testsuites` | list | Test suite identifiers for validation |
| `smell_content` | string | Git diff showing the code smell introduction |
| `gt_content` | string | Git diff showing the ground truth refactoring |
| `hash` | string | Unique hash identifier |
| `commit_hash` | string | Git commit hash of the original code |
| `project_name` | string | Source project name |
| `settings` | dict | Project settings (src_path, env_vars, etc.) |
| `smell_analysis` | string | Detailed analysis of the code smell |
---

### Dataset Statistics

| Metric | Count |
|--------|-------|
| **Total Instances** | 147 |
| **Total Evaluation Cases** | 294 |
| **Code Smell Types** | 7 |
| **Source Projects** | 7 |
| **Difficulty Levels**| 3 |
| **Instruction Types** | 2 (targeted, guided) |

> **Note:** Each instance includes two different instruction types (`hint_targeted` and `hint_guided`), resulting in 147 × 2 = 294 unique evaluation cases.

#### By Code Smell Type

| Type | Count |
|------|-------|
| feature_envy | 21 |
| data_clumps | 21 |
| dead_code_elimination | 21 |
| deeply_inlined_method | 21 |
| god_classes | 21 |
| interface_segregation | 21 |
| shotgun_surgery | 21 |

## Project Structure

```
├── prepare_smell_cases.py    # Main entry point for one-click pipeline
├── smell_benchmark.py        # Core script for smell injection and test validation
├── smell_type.json           # Code smell types and injection strategies
├── repo_list.json            # Repository metadata (URLs, commit IDs, setup commands)
├── testunits.py              # Test utilities for validating injected smells
├── Dockerfile                # Docker configuration for reproducible environment
├── ast_analyzers.py          # AST-based code analysis module
├── harbor_adapter/           # Adapter for Harbor-compatible benchmark format
└── output/
    └── smell_codes.json      # Generated benchmark dataset
```

---

# Benchmark Pipeline

## Option 1: One-Click Reproducibility (Docker Environment)

If you have a Docker environment available, you can use the one-click reproducibility approach:

```bash
python -u prepare_smell_cases.py --agent anthropic/claude-sonnet-4.5
```

This script automatically executes the entire pipeline for all predefined repositories, including:

- Repository cloning
- Environment setup
- Candidate Discovery
- Smell injection
- Quality Verification
- Benchmark construction
---

## Quick Test

To quickly test the overall pipeline without running the full benchmark:

```bash
python -u prepare_smell_cases.py --project-name click --agent mock
```

---

## Option 2: Local Execution (Without Docker)

If you don't have Docker available, follow these steps to run the benchmark locally:

### Step 1: Install Code Agent CLI

Install the code agent CLI of your choice (e.g., Claude Code, Qwen Code):

```bash
# Example: Install Claude Code
npm install -g @anthropic-ai/claude-code
```

### Step 2: Install Python Dependencies

```bash
pip install -r repo_requirements.txt
```

### Step 3: Run Smell Benchmark

Execute the benchmark for each repository:

```bash
# Run for a single repository
python -u smell_benchmark.py --project-name click --agent claude_code/mock
```

### Step 4: Collect All Smell Codes

After running the benchmark for all repositories, collect the results into a single file:

```bash
python -u collect_smell_codes.py
```

This will generate `output/smell_codes.json` containing all smell codes from all repositories.

---

# Evaluation

We support evaluation on [Harbor](https://github.com/harbor-framework/harbor), a framework for benchmarking code agents.

## Generating Harbor-Style Benchmark

After successfully generating `output/smell_codes.json`, follow these steps to create a Harbor-compatible dataset:

### Step 1: Navigate to harbor_adapter

```bash
cd harbor_adapter
```

### Step 2: Generate Harbor Dataset

```bash
python -u run_adapter.py --task-dir <task-dir>
```

This will convert the smell codes into Harbor-style benchmark format in the `task-dir/` directory.

### Step 3: Run Evaluation with Harbor

Clone the Harbor repository and follow the instructions:

```bash
git clone https://github.com/harbor-framework/harbor.git
cd harbor
# Follow Harbor's README to configure and run the evaluation
```

For detailed Harbor configuration and usage, please refer to the [Harbor documentation](https://github.com/harbor-framework/harbor).

---

<!-- # Citation

If you use this benchmark, please cite:

```bibtex
@dataset{smellbench_2026,
  title={SmellBench: A Benchmark for Code Smell Detection and Refactoring},
  year={2026}
}
``` -->