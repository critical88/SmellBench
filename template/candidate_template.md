You are a code analysis expert. Your task is to find methods/classes where it would be **convenient to inject** a specific code smell — NOT to find places that already exhibit this smell.

The goal is to identify code locations where the structure, complexity, and cross-module relationships make it natural and easy to introduce the smell while keeping the code compilable and tests passing.

## Source path: `[SRC_PATH]`

## Eligible files (lines > 500, no utility/helper files)
[ELIGIBLE_FILES]

## Smell type to inject: [SMELL_TYPE]
[SMELL_DESC]
[HINTS]

## Strategy — BE EFFICIENT
Do NOT read every file. Instead:
1. Based on file names and module structure, pick the 3-5 most promising files
2. Use `grep` to quickly locate class definitions, large methods, and cross-module interactions
3. Only read specific sections of files (use line ranges) to verify candidates
4. Prioritize files with core business logic (e.g., core.py, models.py, engine.py) over peripherals

## Requirements
Find exactly 5 candidates. Each candidate should be a method/class where injecting `[SMELL_TYPE]` would be **easy and natural** — meaning the surrounding code structure supports the injection without breaking functionality.

Each candidate needs:
- `file`: relative path from repo root
- `class_name`: class name (null if standalone function)
- `method_name`: method/function name (for god_classes/interface_segregation, can be null)
- `line_number`: the actual starting line number (verify by reading)
- `reason`: 1-2 sentences explaining why this location is a good **injection point** (what structural properties make it easy to introduce the smell here)

**IMPORTANT**: At most 2 candidates may come from the same file. Spread candidates across different files to ensure diversity.

**DIVERSITY REQUIREMENT**: The 5 candidates must be substantially different from each other:
- They should involve different classes/functions with different responsibilities
- They should target different code patterns or architectural concerns
- Avoid picking multiple methods from the same class or methods that do similar things

## Output
After finding all candidates, output a single JSON block:
```json
{
  "[SMELL_TYPE]": [
    {"file": "...", "class_name": "...", "method_name": "...", "line_number": 123, "reason": "..."}
  ]
}
```

The key must be exactly: [SMELL_TYPE]
It must have exactly 5 entries.
