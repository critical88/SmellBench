You previously injected a "[SMELL_TYPE]" code smell into this codebase.
The changes you made caused the following unit tests to FAIL.

⚠️ **CRITICAL**: Your task is to fix the TESTS, NOT to remove the smell!

**Your task**: Fix the code so that all tests pass while **KEEPING the injected code smell intact**.
- The smell must still be present and non-trivial after your fix
- You are fixing **test failures caused by your injection**, NOT refactoring the smell away
- Think of it as: "make my smell injection more robust so it doesn't break tests"

**DO NOT**:
- Remove or refactor the smell you injected
- "Clean up" the code smell
- Make the code better quality — keep the smell as is

**DO**:
- Fix any syntax errors, import errors, or runtime errors
- Adjust test-related code if needed
- Make minimal changes to pass tests while preserving the smell pattern

Do NOT run any tests yourself — testing is handled externally.

## Diff of your previous changes
```diff
[SMELL_CONTENT]
```

## Failing test scripts
[TEST_SCRIPTS]

## Test error output
```
[TEST_ERROR_OUTPUT]
```

## Requirements
1. **Fix the failing tests while preserving the code smell injection** — the smell pattern must remain
2. The code must compile/run correctly
3. **DO NOT remove or refactor the smell** — you are fixing test failures, not improving code quality
4. DO NOT create new files
5. DO NOT run any test commands (pytest, unittest, etc.)
6. Make **minimal changes** — only fix what's broken, keep the smell as you originally injected it

After making your fixes, output the same JSON format as before:
```json
{
  "smell_type": "Type name",
  "hint_targeted": "Natural language task: tell agent to find and refactor the smell. Include smell type + file + class/method. No fixed format.",
  "hint_guided": "Natural language task: tell agent to find and refactor the smell. Include ONLY smell type + the single main file. Do NOT reveal multiple files, class names, or method names. No fixed format.",
  "hint_open": "Natural language task: tell agent to find and refactor code smells in a given file. Include ONLY the file path(s). Do NOT reveal the smell type, class names, or method names. No fixed format.",
  "smell_function": ["absolute/path/to/file", "ClassName", "methodName"],
  "test_functions": [
    ["absolute/path/to/file", "ClassName", "methodName"],
    ["absolute/path/to/other", "ClassName", "methodName"]
  ]
}
```
