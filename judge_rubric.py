"""LLM-as-Judge rubric template for code smell refactoring evaluation.

Dynamically selects smell-type-specific criteria and builds a judge prompt.
Usage:
    python judge_rubric.py --smell-codes output/smell_codes.json --instance-id <id> --agent-diff <path>
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from utils import _run_git_command
from claude_cli import call_llm, extract_json_from_response

# ---------------------------------------------------------------------------
# General evaluation dimensions (always included)
# ---------------------------------------------------------------------------

GENERAL_DIMENSIONS = """### General Evaluation Dimensions (score each 0-10)

1. **Smell Elimination Completeness** (score 0-10)
   - 10: Smell fully eliminated; no residual smell code, unused imports, or orphaned helpers remain
   - 8: Smell substantially eliminated; only minor traces remain
   - 6: Core smell addressed but some related artifacts (unused helpers, stale registrations) left behind
   - 4: Only the most obvious smell location fixed; secondary artifacts untouched
   - 2: Minimal effort; smell barely addressed
   - 0: Smell not addressed or new smell introduced

2. **Cross-File Coordination** (score 0-10)
   - 10: All smell-related code properly addressed; no orphaned imports, dead helpers, or dangling references left behind
   - 8: Nearly all cross-file impacts handled; one minor leftover
   - 6: Core changes correct; some related artifacts (unused helpers, stale imports) remain in other files
   - 4: Some cross-file changes made but several inconsistencies or leftovers
   - 2: Minimal cross-file awareness; related code in other files ignored
   - 0: Cross-file coordination largely missing or incorrect

3. **Structural Soundness** (score 0-10)
   - 10: Proper decomposition; single responsibility; appropriate abstraction
   - 8: Sound structure with minor imperfections
   - 6: Reasonable but some unnecessary complexity
   - 4: Noticeable structural issues; responsibilities not well separated
   - 2: Significant structural problems
   - 0: Introduces new code smells or anti-patterns

4. **Code Quality & Readability** (score 0-10)
   - 10: Clean, idiomatic, well-named, easy to maintain
   - 8: Good quality; minor naming or style improvements possible
   - 6: Acceptable quality; some naming or structural issues
   - 4: Below average; multiple readability concerns
   - 2: Poor quality; hard to follow
   - 0: Significantly worse readability than before"""


# ---------------------------------------------------------------------------
# Difficulty-level guidance (optional, appended when difficulty is provided)
# ---------------------------------------------------------------------------

DIFFICULTY_GUIDANCE = {
    "easy": (
        "**Difficulty: Easy** (1-2 files). "
        "Expect straightforward refactoring. Penalize heavily for missed files."
    ),
    "medium": (
        "**Difficulty: Medium** (2-3 files). "
        "Expect handling of indirect delegation and wrapper patterns. "
        "Be more lenient on edge cases."
    ),
    "hard": (
        "**Difficulty: Hard** (3-4 files). "
        "Expect handling of dynamic dispatch, red herrings, and design patterns. "
        "Focus on whether the agent correctly distinguishes real smells from intentional patterns."
    ),
    "expert": (
        "**Difficulty: Expert** (4-5 files). "
        "Expect handling of dynamic dispatch, red herrings, and design patterns. "
        "Focus on whether the agent correctly distinguishes real smells from intentional patterns."
    ),
}





# ---------------------------------------------------------------------------
# Output format template
# ---------------------------------------------------------------------------

OUTPUT_FORMAT = """## Output Format

Return your evaluation as JSON:
```json
{
  "smell_elimination": {"score": <0-10>, "justification": "<brief>"},
  "cross_file_coordination": {"score": <0-10>, "justification": "<brief>"},
  "structural_soundness": {"score": <0-10>, "justification": "<brief>"},
  "code_quality": {"score": <0-10>, "justification": "<brief>"},
  "summary": "<2-3 sentence overall assessment>"
}
```"""


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

GENERAL_KEYS = [
    "smell_elimination",
    "cross_file_coordination",
    "structural_soundness",
    "code_quality",
]

MAX_SCORE = 10  # each dimension is scored 0-10
FINAL_SCALE = 10  # final score normalized to 0-10 (equal weight across all rubrics)



def prepare_repo_with_smell(
    repo_dir: str,
    repo_name: str,
    smell_diff: str,
    repo_list_path: Optional[str] = None,
) -> None:
    """Reset the repo to its base commit (from repo_list.json), apply smell diff, then commit.

    This sets up the repo so that the code agent can see the smelly code
    in its working directory before judging.
    """
    if repo_list_path is None:
        repo_list_path = str(Path(__file__).resolve().parent / "repo_list.json")
    with open(repo_list_path, "r", encoding="utf-8") as f:
        repo_list = json.load(f)
    if repo_name not in repo_list:
        raise ValueError(
            f"Repo {repo_name!r} not found in {repo_list_path}. "
            f"Available: {', '.join(repo_list.keys())}"
        )
    base_commit = repo_list[repo_name]["commit_id"]
    print(f"  Resetting {repo_dir} to base commit {base_commit}")
    _run_git_command(["reset", "--hard", base_commit], cwd=repo_dir)

    smell_diff = os.path.abspath(smell_diff)
    print(f"  Applying smell diff: {smell_diff}")
    _run_git_command(["apply", smell_diff], cwd=repo_dir)

    print(f"  Staging and committing smell changes")
    _run_git_command(["add", "-A"], cwd=repo_dir)
    _run_git_command(["commit", "-m", "apply smell for judge evaluation"], cwd=repo_dir)


def call_cli_judge(
    prompt: str,
    cwd: str = ".",
) -> Dict[str, Any]:
    """Call the Claude CLI (agent mode) to get an LLM judge evaluation.

    Returns:
        Dict with keys: "parsed" (the parsed JSON result or None),
        "raw" (raw response text), "usage" (token usage dict).
    """
    from claude_cli import call_claude_cli, extract_json_from_response as cli_extract

    raw_text, _trajectory, usage = call_claude_cli(prompt, cwd=cwd)
    parsed = cli_extract(raw_text)
    return {"parsed": parsed, "raw": raw_text, "usage": usage}


def compute_weighted_score(result: Dict[str, Any]) -> float:
    """Compute an equally-weighted score normalized to 0-55.

    All rubric dimensions (general) are treated equally.
    Final score = (average of all 0-10 scores) / 10 * 55.
    """
    all_scores: list[float] = []

    # General dimensions
    for key in GENERAL_KEYS:
        dim = result.get(key, {})
        if isinstance(dim, dict) and "score" in dim:
            all_scores.append(dim["score"])

    
    if not all_scores:
        return 0.0

    avg = sum(all_scores) / len(all_scores)
    final = (avg / MAX_SCORE) * FINAL_SCALE
    return round(final, 2)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------



def extract_affected_files_from_diff(diff_text: str) -> list[str]:
    """Extract the list of affected file paths from a unified diff."""
    files: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                continue
            if path.startswith("b/"):
                path = path[2:]
            files.append(path)
    return files


def extract_smelly_code_from_diff(diff_text: str) -> str:
    """Extract the 'after' state (smelly code) from a unified diff.

    Keeps context lines and '+' lines (stripping the '+' prefix),
    skips '-' lines and diff metadata. Groups by file.
    """
    chunks: list[str] = []
    current_file: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if current_file != path:
                current_file = path
                chunks.append(f"# File: {path}")
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("-"):
            # removed line — not part of the smelly code
            continue
        if line.startswith("+"):
            chunks.append(line[1:])  # strip '+' prefix
        else:
            # context line (unchanged)
            chunks.append(line[1:] if line.startswith(" ") else line)

    return "\n".join(chunks)


def build_judge_prompt(
    smell_type: str,
    refactored_code: str,
    smell_analysis: Optional[str] = None,
    difficulty: Optional[str] = None,
    label: str = "Candidate",
) -> str:
    """Build an LLM-as-judge prompt for evaluating a single refactored version.

    Args:
        smell_type: The code smell type (e.g. "feature_envy").
        refactored_code: The refactored code (diff) to evaluate.
        smell_analysis: Pre-generated analysis of the smell (from Step 1).
        difficulty: Optional difficulty level ("hard", "expert").
        label: Label for the refactored code (e.g. "Ground Truth", "Agent").

    Returns:
        A formatted prompt string ready to send to an LLM judge.
    """
    # Build difficulty guidance
    diff_section = ""
    if difficulty:
        guidance = DIFFICULTY_GUIDANCE.get(difficulty.lower().strip())
        if guidance:
            diff_section = f"\n\n### Difficulty Context\n{guidance}"

    # Build smell analysis section
    analysis_section = ""
    if smell_analysis:
        analysis_section = f"""
### Expert Analysis of the Smell
The following analysis describes the smell that was introduced, its root cause, and what matters most when evaluating whether a fix truly addresses the problem. Use it to inform your scoring — but judge the refactoring on its own merits.

{smell_analysis}
"""

    # Build dynamic output format
    output_format = f"""## Output Format

Return your evaluation as JSON:
```json
{{
  "smell_elimination": {{"score": <0-10>, "justification": "<brief>"}},
  "cross_file_coordination": {{"score": <0-10>, "justification": "<brief>"}},
  "structural_soundness": {{"score": <0-10>, "justification": "<brief>"}},
  "code_quality": {{"score": <0-10>, "justification": "<brief>"}},
  "summary": "<2-3 sentence overall assessment>"
}}
```"""

    prompt = f"""You are an expert code reviewer evaluating a refactored version of code that originally contained a "{smell_type}" code smell.

## Context
- **Smell Type**: {smell_type}
{analysis_section}
**IMPORTANT**: The refactoring below is provided in **unified diff format** (git diff output). Lines starting with `-` are removed, lines starting with `+` are added, and context lines are unchanged. Evaluate the *intent and quality of the changes*, not the completeness of the code shown — diffs only show changed regions, not the full files.

### {label} Refactoring (diff fixing the smell)
```diff
{refactored_code}
```

## Evaluation Rubric

{GENERAL_DIMENSIONS}

{diff_section}

{output_format}"""

    return prompt


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def evaluate_instance(
    instance: Dict[str, Any],
    agent_diff: str,
    backend: str = "llm",
    model: str = "claude-sonnet-4-5-20250929",
    base_url: Optional[str] = None,
    cwd: str = ".",
    repo_dir: str = "../project",
) -> Dict[str, Any]:
    """Evaluate a single diff against a smell instance.

    This function evaluates only once — the provided agent_diff.
    To evaluate ground truth, pass instance["gt_content"] as agent_diff.

    Args:
        instance: A single entry from smell_codes.json.
        agent_diff: The diff content to evaluate (string, not file path).
        backend: "llm" for Anthropic API, "cli" for Claude CLI agent.
        model: Model name for LLM backend.
        base_url: Optional API base URL override.
        cwd: Working directory for CLI backend.
        repo_dir: Root project repo directory for CLI backend.

    Returns:
        Dict with weighted_score, general, summary, usage.

    Raises:
        ValueError: If instance is missing required fields.
    """
    import sys

    # Extract fields from instance
    smell_type = instance.get("type", "")
    difficulty = instance.get("difficulty")
    smell_analysis = instance.get("smell_analysis")
    repo_name = instance.get("project_name", "")
    instance_id = instance.get("instance_id", "")

    # Validate required fields
    missing = []
    if not smell_type:
        missing.append("type")
    if not smell_analysis:
        missing.append("smell_analysis")
    if missing:
        raise ValueError(
            f"Instance {instance_id!r} is missing required fields: {', '.join(missing)}"
        )

    # For cli backend: reset repo and apply smell so the agent sees smelly code
    smell_content = instance.get("smell_content", "")
    if backend == "cli" and repo_name and smell_content:
        import tempfile
        full_repo_dir = os.path.join(repo_dir, repo_name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as tmp:
            tmp.write(smell_content)
            tmp_diff_path = tmp.name
        try:
            print(f"Preparing repo: reset {repo_name} to base commit and apply smell")
            prepare_repo_with_smell(full_repo_dir, repo_name, tmp_diff_path)
        finally:
            os.unlink(tmp_diff_path)

    # Evaluate the provided diff
    print(f"Evaluating {smell_type} [{instance_id}]...")
    prompt = build_judge_prompt(
        smell_type=smell_type,
        refactored_code=agent_diff,
        smell_analysis=smell_analysis,
        difficulty=difficulty,
    )
    if backend == "llm":
        llm_result = call_llm(prompt, model=model, base_url=base_url)
    else:
        llm_result = call_cli_judge(prompt, cwd=cwd)

    parsed = llm_result["parsed"]
    if parsed is None:
        print(f"Failed to parse JSON from judge response.", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(llm_result["raw"], file=sys.stderr)
        raise RuntimeError(f"Failed to parse judge response for instance {instance_id!r}")

    parsed["weighted_score"] = compute_weighted_score(parsed)

    # Organize scores by category
    general_scores = {}
    for k in GENERAL_KEYS:
        if k in parsed:
            general_scores[k] = parsed[k]

    result = {
        "instance_id": instance_id,
        "weighted_score": parsed["weighted_score"],
        "general": general_scores,
        "summary": parsed.get("summary", ""),
        "usage": llm_result["usage"],
        "_meta": {
            "smell_type": smell_type,
            "difficulty": difficulty,
            "backend": backend,
            "model": model if backend == "llm" else None,
        },
    }
    print(f"  weighted_score: {parsed['weighted_score']}")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge evaluation on a single refactoring diff."
    )
    parser.add_argument("--smell-codes", default="output/smell_codes.json",
                        help="Path to smell_codes.json containing case instances.")
    parser.add_argument("--instance-id", required=True,
                        help="instance_id of the case to evaluate.")
    parser.add_argument("--agent-diff", required=True,
                        help="Path to diff file to evaluate (agent diff or ground truth diff).")
    parser.add_argument("--backend", choices=["llm", "cli"], default="llm",
                        help="Judge backend: 'llm' uses Anthropic API directly, 'cli' uses Claude CLI agent (default: llm).")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929",
                        help="Model to use for LLM judge (default: claude-sonnet-4-5-20250929). Only used with --backend llm.")
    parser.add_argument("--base-url", default=None,
                        help="Base URL for Anthropic API (overrides ANTHROPIC_BASE_URL env var). Only used with --backend llm.")
    parser.add_argument("--cwd", default=".", help="Working directory for Claude CLI call. Only used with --backend cli.")
    parser.add_argument("--repo-dir", default="../project",
                        help="Path to the project repo. Used with --backend cli to reset and apply smell before judging.")
    parser.add_argument("--output", "-o", default=None, help="Path to save judge result JSON.")
    args = parser.parse_args()

    # Load instance from smell_codes.json
    with open(args.smell_codes, "r", encoding="utf-8") as f:
        all_instances = json.load(f)

    instance = None
    for entry in all_instances:
        if entry.get("instance_id") == args.instance_id:
            instance = entry
            break
    if instance is None:
        print(f"Instance {args.instance_id!r} not found in {args.smell_codes}", file=sys.stderr)
        sys.exit(1)

    # Read diff file
    with open(args.agent_diff, "r", encoding="utf-8") as f:
        agent_diff = f.read()

    # Run evaluation
    try:
        result = evaluate_instance(
            instance=instance,
            agent_diff=agent_diff,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            cwd=args.cwd,
            repo_dir=args.repo_dir,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Result saved to {args.output}")
    else:
        print(output_json)

    print(f"\n{'='*50}")
    print(f"  Score: {result['weighted_score']} / 10")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
