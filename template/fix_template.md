You previously injected a "[SMELL_TYPE]" code smell into this codebase.
The changes you made caused the following unit tests to FAIL.

Your task: fix the code so that all tests pass while KEEPING the injected code smell intact.
The smell must still be present and non-trivial after your fix.

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
1. Fix the failing tests while preserving the code smell injection
2. The code must compile/run correctly
3. Do NOT remove the smell — only fix the breakage
4. Do NOT create new files
5. Do NOT run any test commands (pytest, unittest, etc.)

After making your fixes, output the same JSON format as before:
```json
{
  "hint_targeted": "Natural language task: tell agent to find and refactor the smell. Include smell type + file + class/method. No fixed format.",
  "hint_guided": "Natural language task: tell agent to find and refactor the smell. Include ONLY smell type + the single main file. Do NOT reveal multiple files, class names, or method names. No fixed format.",
  "hint_open": "Natural language task: tell agent to find and refactor code smells in a given file. Include ONLY the file path(s). Do NOT reveal the smell type, class names, or method names. No fixed format.",
  "smell_function": ["<absolute_file_path>", "<class name or null>", "<function name or null>"],
  "test_functions": [["<absolute_file_path>", "<class name or null>", "<function_name>"]]
}
```
