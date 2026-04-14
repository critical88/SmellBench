"""LLM-as-Judge rubric template for code smell refactoring evaluation.

Dynamically selects smell-type-specific criteria and builds a judge prompt.
Usage:
    from judge_rubric import build_judge_prompt
    prompt = build_judge_prompt("feature envy", original_code, ground_truth, agent_code)
"""

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# General evaluation dimensions (always included)
# ---------------------------------------------------------------------------

GENERAL_DIMENSIONS = """### General Evaluation Dimensions (score each 0-10)

1. **Smell Elimination Completeness** (Weight: 30%)
   - 10: Smell fully eliminated; no residual traces
   - 6: Partially eliminated; core issue addressed but secondary aspects remain
   - 0: Smell not addressed or new smell introduced

2. **Code Quality & Readability** (Weight: 25%)
   - 10: Clean, idiomatic, well-named, easy to maintain
   - 6: Acceptable quality; some naming or structural issues
   - 0: Significantly worse readability than before

3. **Structural Soundness** (Weight: 25%)
   - 10: Proper decomposition; single responsibility; appropriate abstraction
   - 6: Reasonable but some unnecessary complexity
   - 0: Introduces new code smells or anti-patterns

4. **Cross-File Coordination** (Weight: 20%)
   - 10: All affected files correctly modified; imports and call sites consistent
   - 6: Core cross-file changes correct; some call sites or imports missed
   - 0: Cross-file coordination largely missing or incorrect"""


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
# Smell-type-specific rubrics
# ---------------------------------------------------------------------------

RUBRICS: Dict[str, Dict[str, Any]] = {
    "feature_envy": {
        "description": "A function that is more interested in data from other classes than its own, indicating misplaced behavior.",
        "focus": "Whether the envious method is moved to the class whose data it primarily accesses, and whether data locality is improved.",
        "criteria": [
            {
                "name": "Method Placement",
                "excellent": "Method moved to the correct class that owns the data",
                "acceptable": "Method partially restructured but still accesses foreign data",
                "poor": "Method remains in the wrong class",
            },
            {
                "name": "Data Locality",
                "excellent": "After refactoring, the method operates on data within its own class",
                "acceptable": "Most data accesses are local but some foreign accesses remain",
                "poor": "Method still primarily accesses data from other classes",
            },
            {
                "name": "Delegation Appropriateness",
                "excellent": "If delegation is used instead of moving, the delegation pattern is clean and justified",
                "acceptable": "Delegation works but introduces minor indirection overhead",
                "poor": "Delegation is awkward or hides the original coupling without fixing it",
            },
        ],
    },
    "god_classes": {
        "description": "A class that centralizes too much functionality, violating single responsibility and becoming hard to maintain.",
        "focus": "Whether distinct responsibilities are correctly identified and extracted into separate, cohesive classes.",
        "criteria": [
            {
                "name": "Responsibility Identification",
                "excellent": "All distinct responsibilities correctly identified and separated",
                "acceptable": "Main responsibilities identified but boundaries imprecise",
                "poor": "Responsibilities not clearly separated",
            },
            {
                "name": "Extracted Class Cohesion",
                "excellent": "Each new class has a single clear responsibility",
                "acceptable": "New classes are somewhat cohesive",
                "poor": "New classes are still too broad or too narrow",
            },
            {
                "name": "State Management",
                "excellent": "Shared state properly encapsulated; no mutable state leaks",
                "acceptable": "Minor state management issues",
                "poor": "State scattered or leaked between classes",
            },
        ],
    },
    "data_clumps": {
        "description": "Groups of variables that are frequently passed together, suggesting poor encapsulation or a missing class abstraction.",
        "focus": "Whether all instances of the data clump are identified across files and replaced with a well-designed abstraction.",
        "criteria": [
            {
                "name": "Clump Completeness",
                "excellent": "All instances of the data clump identified and refactored",
                "acceptable": "Most instances refactored; a few call sites missed",
                "poor": "Only partial instances refactored",
            },
            {
                "name": "Abstraction Design",
                "excellent": "Well-named class/dataclass with appropriate fields and optional methods",
                "acceptable": "Reasonable grouping but naming or field design could be better",
                "poor": "Overly generic or poorly named grouping",
            },
            {
                "name": "Field Boundary Accuracy",
                "excellent": "Exactly the right parameters grouped — no missing or extra fields",
                "acceptable": "Core fields correct but includes one unnecessary or misses one relevant field",
                "poor": "Grouping is wrong — key parameters excluded or unrelated ones included",
            },
        ],
    },
    "shotgun_surgery": {
        "description": "A change that requires making small modifications in many different classes or files, indicating scattered responsibilities.",
        "focus": "Whether scattered logic is properly consolidated into a single location so that a conceptual change requires modifying only one place.",
        "criteria": [
            {
                "name": "Fragment Discovery",
                "excellent": "All scattered fragments identified including indirect ones",
                "acceptable": "Most direct fragments found; some indirect ones missed",
                "poor": "Significant fragments missed",
            },
            {
                "name": "Consolidation Strategy",
                "excellent": "Logic centralized in a single, well-chosen location",
                "acceptable": "Logic partially centralized; some scatter remains",
                "poor": "No meaningful consolidation",
            },
            {
                "name": "Abstraction Appropriateness",
                "excellent": "Unified logic uses an appropriate encapsulation (function/class/config) without over-engineering",
                "acceptable": "Encapsulation works but is slightly over- or under-engineered",
                "poor": "No proper abstraction or introduces unnecessary complexity",
            },
        ],
    },
    "dead_code_elimination": {
        "description": "Code that is never executed or used, increasing complexity and maintenance burden.",
        "focus": "Whether dead code is accurately identified and completely removed without leaving orphaned references.",
        "criteria": [
            {
                "name": "Dead Code Coverage",
                "excellent": "All dead code identified and removed",
                "acceptable": "Most dead code removed; some unreachable paths remain",
                "poor": "Significant dead code left in place",
            },
            {
                "name": "Import/Reference Cleanup",
                "excellent": "All orphaned imports and references cleaned up",
                "acceptable": "Most imports cleaned; some dangling references",
                "poor": "Leaves broken or unnecessary imports",
            },
            {
                "name": "Red Herring Avoidance",
                "excellent": "Correctly preserves code that looks dead but is actually reachable",
                "acceptable": "Mostly correct but uncertain about one borderline case",
                "poor": "Removes live code that appears dead, or keeps obvious dead code out of caution",
            },
        ],
    },
    "interface_segregation": {
        "description": "Interfaces that are too large, forcing implementations to depend on methods they don't use.",
        "focus": "Whether the fat interface is correctly split into focused, cohesive interfaces and unnecessary stubs are removed.",
        "criteria": [
            {
                "name": "Interface Decomposition",
                "excellent": "Interfaces split along cohesive responsibility boundaries",
                "acceptable": "Interfaces split but boundaries not ideal",
                "poor": "Interfaces not meaningfully separated",
            },
            {
                "name": "Stub Elimination",
                "excellent": "All unnecessary stub/pass implementations removed",
                "acceptable": "Most stubs removed; some remain",
                "poor": "Stubs still present in implementors",
            },
            {
                "name": "Interface Granularity",
                "excellent": "Split granularity is appropriate — each interface is focused but not overly narrow",
                "acceptable": "Slightly too coarse or too fine-grained",
                "poor": "Interfaces still too fat, or shattered into trivial single-method fragments",
            },
        ],
    },
    "deeply_inlined_method": {
        "description": "A method whose sub-method implementations are copied into itself, creating extreme complexity.",
        "focus": "Whether inlined code fragments are correctly identified and extracted back into well-scoped methods at the right abstraction level.",
        "criteria": [
            {
                "name": "Fragment Identification",
                "excellent": "All inlined fragments correctly identified despite variable renames and restructuring",
                "acceptable": "Major fragments identified; some minor ones missed",
                "poor": "Fragments not correctly identified",
            },
            {
                "name": "Extraction Granularity",
                "excellent": "Each extracted method has a clear, single purpose at the right abstraction level",
                "acceptable": "Extraction is reasonable but some methods too large or too small",
                "poor": "Over-extraction (too many tiny methods) or under-extraction",
            },
            {
                "name": "Depth Handling",
                "excellent": "All inlining levels correctly unwound (depth 1, 2, 3+)",
                "acceptable": "Top-level inlining resolved; deeper levels partially addressed",
                "poor": "Only surface-level extraction; deep inlining remains",
            },
        ],
    },
}



# ---------------------------------------------------------------------------
# Output format template
# ---------------------------------------------------------------------------

OUTPUT_FORMAT = """## Output Format

Return your evaluation as JSON:
```json
{
  "smell_elimination": {"score": <0-10>, "justification": "<brief>"},
  "code_quality": {"score": <0-10>, "justification": "<brief>"},
  "structural_soundness": {"score": <0-10>, "justification": "<brief>"},
  "cross_file_coordination": {"score": <0-10>, "justification": "<brief>"},
  "smell_specific": {
    "<criterion_name>": {"score": <0-10>, "justification": "<brief>"},
    ...
  },
  "summary": "<2-3 sentence overall assessment>"
}
```"""


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

GENERAL_KEYS = [
    "smell_elimination",
    "code_quality",
    "structural_soundness",
    "cross_file_coordination",
]

# Weights within the general group (must sum to 1.0)
GENERAL_WEIGHTS = {
    "smell_elimination": 0.30,
    "code_quality": 0.25,
    "structural_soundness": 0.25,
    "cross_file_coordination": 0.20,
}

# Proportion between general and smell-specific groups
GENERAL_RATIO = 0.60
SPECIFIC_RATIO = 0.40

MAX_SCORE = 10  # each dimension is scored 0-10
FINAL_SCALE = 10  # final score normalized to 0-10


def compute_weighted_score(result: Dict[str, Any]) -> float:
    """Compute a weighted score normalized to 0-10.

    General dimensions (60%): weighted by GENERAL_WEIGHTS, each scored 0-10.
    Smell-specific dimensions (40%): equally weighted across 3 criteria, each scored 0-10.
    """
    # General: weighted average of 0-5 scores, then normalize to 0-1
    general_score = 0.0
    for key in GENERAL_KEYS:
        dim = result.get(key, {})
        score = dim.get("score", 0) if isinstance(dim, dict) else 0
        general_score += score * GENERAL_WEIGHTS.get(key, 0)
    general_norm = general_score / MAX_SCORE  # 0-1

    # Smell-specific: simple average of 0-5 scores, then normalize to 0-1
    smell_specific = result.get("smell_specific", {})
    specific_scores = [
        v["score"] for v in smell_specific.values()
        if isinstance(v, dict) and "score" in v
    ]
    if specific_scores:
        specific_norm = sum(specific_scores) / (len(specific_scores) * MAX_SCORE)
    else:
        specific_norm = 0.0

    final = (GENERAL_RATIO * general_norm + SPECIFIC_RATIO * specific_norm) * FINAL_SCALE
    return round(final, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _format_criteria(criteria: list) -> str:
    """Format smell-specific criteria into readable text."""
    lines = []
    for i, c in enumerate(criteria, 1):
        lines.append(f"{i}. **{c['name']}**")
        lines.append(f"   - 10 (Excellent): {c['excellent']}")
        lines.append(f"   - 6 (Acceptable): {c['acceptable']}")
        lines.append(f"   - 0 (Poor): {c['poor']}")
    return "\n".join(lines)


def get_rubric(smell_type: str) -> Optional[Dict[str, Any]]:
    """Look up the rubric for a given smell type. Returns None if not found."""
    return RUBRICS.get(smell_type.lower().strip().replace(" ", "_"))


def build_judge_prompt(
    smell_type: str,
    original_code: str,
    refactored_code: str,
    difficulty: Optional[str] = None,
    label: str = "Candidate",
) -> str:
    """Build an LLM-as-judge prompt for evaluating a single refactored version.

    Args:
        smell_type: The code smell type (e.g. "feature_envy").
        original_code: The original code containing the smell.
        refactored_code: The refactored code to evaluate.
        difficulty: Optional difficulty level ("hard", "expert").
        label: Label for the refactored code (e.g. "Ground Truth", "Agent").

    Returns:
        A formatted prompt string ready to send to an LLM judge.

    Raises:
        ValueError: If the smell type is not recognized.
    """
    rubric = get_rubric(smell_type)
    if rubric is None:
        available = sorted(RUBRICS.keys())
        raise ValueError(
            f"Unknown smell type {smell_type!r}. "
            f"Available types: {', '.join(available)}"
        )

    # Build smell-specific section
    smell_section = (
        f"### Smell-Specific Criteria for \"{smell_type}\"\n\n"
        f"**Focus:** {rubric['focus']}\n\n"
        f"{_format_criteria(rubric['criteria'])}"
    )

    # Build difficulty guidance
    diff_section = ""
    if difficulty:
        guidance = DIFFICULTY_GUIDANCE.get(difficulty.lower().strip())
        if guidance:
            diff_section = f"\n\n### Difficulty Context\n{guidance}"

    prompt = f"""You are an expert code reviewer evaluating a refactored version of code that originally contained a "{smell_type}" code smell.

## Context
- **Smell Type**: {smell_type}
- **Smell Description**: {rubric['description']}

### Original Code (with smell)
```
{original_code}
```

### {label} Refactoring
```
{refactored_code}
```

## Evaluation Rubric

{GENERAL_DIMENSIONS}

{smell_section}{diff_section}

{OUTPUT_FORMAT}"""

    return prompt


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json
    import sys

    from claude_cli import call_claude_cli, extract_json_from_response

    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge evaluation on smell refactoring diffs (evaluated independently)."
    )
    parser.add_argument("--smell-type", required=True, help="Code smell type (e.g. feature_envy).")
    parser.add_argument("--gt-diff", required=True, help="Path to ground truth diff file.")
    parser.add_argument("--agent-diff", required=True, help="Path to agent diff file.")
    parser.add_argument("--original-diff", default="", help="Path to original (smelly) diff file.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                        help="Difficulty level for evaluation context.")
    parser.add_argument("--cwd", default=".", help="Working directory for claude CLI call.")
    parser.add_argument("--output", "-o", default=None, help="Path to save judge result JSON.")
    args = parser.parse_args()

    # Read diff files
    with open(args.gt_diff, "r", encoding="utf-8") as f:
        gt_code = f.read()
    with open(args.agent_diff, "r", encoding="utf-8") as f:
        agent_code = f.read()

    original_code = ""
    if args.original_diff:
        with open(args.original_diff, "r", encoding="utf-8") as f:
            original_code = f.read()

    # Evaluate ground truth and agent independently
    results = {}
    for label, code, diff_path in [
        ("ground_truth", gt_code, args.gt_diff),
        ("agent", agent_code, args.agent_diff),
    ]:
        print(f"Evaluating {args.smell_type} [{label}]: {diff_path}")
        prompt = build_judge_prompt(
            smell_type=args.smell_type,
            original_code=original_code,
            refactored_code=code,
            difficulty=args.difficulty,
            label=label,
        )
        response_text, trajectory, usage = call_claude_cli(prompt, cwd=args.cwd)
        parsed = extract_json_from_response(response_text)
        if parsed is None:
            print(f"Failed to parse JSON from judge response for [{label}].", file=sys.stderr)
            print("Raw response:", file=sys.stderr)
            print(response_text, file=sys.stderr)
            sys.exit(1)
        parsed["_usage"] = usage
        # Compute weighted score from dimension scores (don't trust LLM's self-calculation)
        parsed["weighted_score"] = compute_weighted_score(parsed)
        results[label] = parsed
        print(f"  [{label}] weighted_score: {parsed['weighted_score']}")

    # Determine verdict by comparing scores
    gt_score = results["ground_truth"]["weighted_score"]
    agent_score = results["agent"]["weighted_score"]
    diff = agent_score - gt_score
    if diff > 0.5:
        verdict = "agent_better"
    elif diff < -0.5:
        verdict = "ground_truth_better"
    else:
        verdict = "comparable"

    final = {
        "ground_truth": results["ground_truth"],
        "agent": results["agent"],
        "verdict": verdict,
        "score_diff": round(diff, 4),
        "_meta": {
            "smell_type": args.smell_type,
            "difficulty": args.difficulty,
            "gt_diff": args.gt_diff,
            "agent_diff": args.agent_diff,
        },
    }

    output_json = json.dumps(final, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Result saved to {args.output}")
    else:
        print(output_json)

    # Print final summary
    print(f"\n{'='*50}")
    print(f"  Ground Truth score: {gt_score}")
    print(f"  Agent score:        {agent_score}")
    print(f"  Diff (agent - gt):  {round(diff, 4)}")
    print(f"  Verdict:            {verdict}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
