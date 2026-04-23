You are a senior software engineer specializing in code transformation and refactoring benchmarks.

Your task is NOT to improve the code. Instead, you must inject a specific type of code smell into the given codebase.

### 🎯 Objective
Inject the following code smell:
[SMELL_TYPE]

into the target location:
[PROJECT_PATH]

[SMELL_TYPE_DESC]

### 📂 Target Candidate (MANDATORY)
You MUST inject the smell starting from the following candidate:
**[TARGET_FILE]** ([TARGET_FILE_LINES] lines) — `[TARGET_CLASS_METHOD]` (line [TARGET_LINE_NUMBER])

- The primary smell logic MUST originate from this candidate method/class
- You may modify related files as needed to satisfy cross-file coupling requirements
- Do NOT ignore this candidate and pick a different starting point

### 📊 Difficulty Level: [DIFFICULTY]
[DIFFICULTY_DESC]

### 📌 Constraints (VERY IMPORTANT)
1. The injected code MUST:
   - Preserve original functionality (behaviorally equivalent)
   - Compile/run correctly
   - Not introduce syntax errors
   - Not create new files
   - **Do NOT run any unit tests or test commands (e.g., pytest, unittest). Testing will be handled separately after your changes are captured.**

2. The smell MUST:
   - Be non-trivial and hard to detect
   - Be deeply integrated (not superficial or easily removable)
   - Require non-local reasoning to refactor
   - **Span across at least [MIN_FILES] files**
   - **Have no single file containing the complete logic**

3. Injection strategy MUST:
   - Prefer structural transformation over simple addition
   - Avoid obvious patterns (e.g., simple duplication or dead code)
   - Increase nesting / coupling / hidden dependencies
   - **Distribute logic across files via indirect calls or shared state**

4. Ensure:
   - The solution to remove the smell is UNIQUE (no multiple trivial fixes)
   - The smell spans across multiple functions or scopes if possible
   - **Local (single-file) refactoring must be insufficient to fix the issue**

[DIFFICULTY_AMPLIFIERS]

---

### 📤 Output Requirements (VERY IMPORTANT)

After generating the modified codebase, you MUST append a JSON object at the very end of your response.

Requirements:
- The JSON must be valid
- Do NOT include any extra text after the JSON
- Do NOT wrap JSON inside explanations
- Place JSON AFTER all code
- All fields MUST be written in English

## CRITICAL: Write DIVERSE, VARIED instructions

Each hint should feel UNIQUE and NATURAL. Use different verbs (refactor/clean up/eliminate/remove/address/fix/resolve/improve/simplify/reorganize/restructure/extract/consolidate) and different sentence structures (imperative/request/question/observation/task assignment).

Format:
```json
{
  "hint_targeted": "A concise, professional task description (1-2 sentences) that specifies the smell type name (e.g., 'dead code', 'god class') and exact location (file, class, method). Use VARIED phrasing. Can mention smell type NAME, but MUST NOT describe: (1) WHY the smell exists or what caused it, (2) HOW the smell manifests (specific pattern/structure), (3) implementation details (how code works internally), (4) mechanisms (technical approach/pattern), (5) architecture (system design/relationships), (6) what the code does (functionality/purpose), (7) multiple affected files.",
  "hint_guided": "A concise, professional task description (1-2 sentences) that specifies the smell type name (e.g., 'dead code', 'god class') and the single main file. Use VARIED phrasing. Can mention smell type NAME, but MUST NOT describe: (1) WHY the smell exists or what caused it, (2) HOW the smell manifests (specific pattern/structure), (3) other files beyond the main file, (4) class/method names (agent discovers these), (5) line numbers, (6) implementation details.",
  "hint_open": "A concise, professional task description (1-2 sentences) asking to refactor the specified file(s). MUST be COMPLETELY GENERIC - only file path + generic refactoring request. Should clearly indicate it's a refactoring task, not a vague review or check. Generic quality terms are ALLOWED (maintainability, readability, code quality, better practices, cleaner code). MUST NOT: (1) hint at SPECIFIC problem type (design issues, responsibilities, encapsulation, coupling, cohesion, complexity, duplication, dependencies, abstraction, inheritance, modularity, separation of concerns), (2) reveal smell type (dead code, god class, feature envy, etc.), (3) mention class/method/function names, (4) mention line numbers, (5) describe what specific issues exist. Key principle: reject hints that reveal WHAT the problem is, not generic quality improvement language.",
  "smell_function": ["<absolute_file_path>", "<class name or null>", "<function name or null>"],
  "test_functions": [
    ["<absolute_file_path>", "<class name or null>", "<function_name>"],
    ["<absolute_file_path>", "<class name or null>", "<function_name>"]
  ]
}
```

### smell_function rules (CRITICAL):
- This identifies the single **most central location** where the smell is centered.
- For class-level smells (e.g., god_classes, interface_segregation), use `[absolute_file_path, class_name, null]` to indicate the class itself.
- For function-level smells, use `[absolute_file_path, class_name_or_null, function_name]`.
- Format: `[absolute_file_path, class_name_or_null, function_name_or_null]` (NOT a nested array).

### test_functions rules (CRITICAL):
- **IMPORTANT**: List **SOURCE CODE functions/methods that were MODIFIED** by the smell injection, NOT test case functions.
- These should be functions in the **production code** , NOT in test files .
- Each entry MUST be at **method/function level** — never a class or module.
- For class-level smells (e.g., god_classes, interface_segregation), list the specific **methods in the source code that were modified** or whose implementations are affected by the injection.
- The call chain of each listed function should **cover the injected changes** — i.e., exercising these functions should trigger the smell-related code paths.
- If the injection modifies multiple methods across files, list **all of them** from the source code — the more coverage the better.
- Format: each entry is `[absolute_file_path, class_name_or_null, function_name]` where file_path points to SOURCE CODE files.
- Example: `["src/jinja2/runtime.py", null, "new_context"]` NOT `["tests/test_api.py", "TestClass", "test_something"]`
