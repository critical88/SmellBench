"""
Mock Agent Client for testing smell benchmark pipeline without real API calls.

This module provides a MockAgentClient that applies real case diffs from examples/
directory, ensuring that tests pass while avoiding API costs.
"""

import os
import json
import subprocess
import tempfile
from typing import Tuple, Optional
from client import AgentClient, AgentResponse


class MockAgentClient(AgentClient):
    """Mock agent for testing without real API calls.

    Returns pre-defined example cases that pass all validation checks.
    Applies real diffs from examples/ directory.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model or "mock-agent-v1"
        self.call_count = 0

    def chat(self, prompt, model=None, *args, **kwargs) -> AgentResponse:
        """Override chat to inject mock file changes."""
        if 'project_repo' not in kwargs:
            raise ValueError("agent chat method must input 'project_repo'")

        project_repo = kwargs['project_repo']
        model = model or self.model

        # Get mock response text
        response_text = self.send_request(prompt, model, cwd=project_repo)

        # Convert to AgentResponse
        response = self._tackle_output_to_response(model, response_text)

        self._record_token_usage(response)

        return response

    def send_request(self, prompt, model=None, cwd=None):
        """Return mock response based on prompt type."""
        self.call_count += 1

        # Detect prompt type
        is_fix_prompt = "test failures" in prompt.lower() or "fix the following" in prompt.lower()
        is_analysis_prompt = (
            "analyzing a code change" in prompt.lower() or
            "analyze this diff" in prompt.lower() or
            "<rubric>" in prompt.lower() or
            "output format" in prompt.lower() and "<analysis>" in prompt.lower()
        )

        if is_fix_prompt:
            # Return a simple fix response
            return self._get_fix_response()
        elif is_analysis_prompt:
            # Return smell analysis response with rubrics
            return self._get_analysis_response()
        else:
            # Return smell injection response
            return self._get_injection_response(cwd)

    def _get_injection_response(self, cwd=None):
        """Generate a realistic smell injection response with mock diff.

        Returns a JSON response that references the actual files modified.
        Loads real case examples from examples/ directory.
        """
        # Detect project language
        language = self._detect_language(cwd) if cwd else "python"

        # Load example case based on language
        examples_dir = os.path.join(os.path.dirname(__file__), "examples")
        example_files = {
            "python": "click_case.json",
            "java": "commons-io_case.json",
            "go": "click_case.json"  # Fallback to python for now
        }

        example_file = os.path.join(examples_dir, example_files.get(language, "click_case.json"))

        if not os.path.exists(example_file):
            return self._get_fallback_injection_response(cwd)

        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                example = json.load(f)

            # Inject mock changes in the repository
            injected_path, class_name, method_name = self._inject_mock_changes(cwd)

            # Use real example but update paths to match injected location if successful
            if injected_path:
                example["smell_function"] = [injected_path, class_name or "MockDataProcessor", method_name or "process_with_many_params"]
                # Update test_functions to point to the same file
                example["test_functions"] = [[injected_path, class_name or "MockDataProcessor", method_name or "process_with_many_params"]]

            # Return the example as JSON
            response = f"""I've successfully injected a {example.get('smell_type', 'code smell')} into the codebase.

```json
{json.dumps(example, indent=2)}
```

The smell has been successfully injected and is ready for testing."""
            return response

        except Exception as e:
            print(f"Warning: Failed to load example case: {e}")
            return self._get_fallback_injection_response(cwd)

    def _inject_mock_changes(self, repo_path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Apply real case diff to create realistic code changes.

        Instead of generating fake code, this applies the actual diff from
        a real successful case to ensure tests pass.

        Returns:
            (target_file, target_class, target_method) tuple
        """
        # Load the real case template
        language = self._detect_language(repo_path) if repo_path else "python"
        examples_dir = os.path.join(os.path.dirname(__file__), "examples")
        example_files = {
            "python": "click_case.json",
            "java": "commons-io_case.json",
            "go": "click_case.json"
        }

        example_file = os.path.join(examples_dir, example_files.get(language, "click_case.json"))

        target_file = None
        target_class = None
        target_method = None

        # Load the real case and apply its diff
        if os.path.exists(example_file):
            try:
                with open(example_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                    # Get the main function info
                    main_function = case_data.get("main_function", [])
                    if main_function and len(main_function) > 0:
                        first_function = main_function[0]
                        if len(first_function) >= 3:
                            file_path = first_function[0]
                            target_class = first_function[1]
                            target_method = first_function[2]
                            target_file = file_path

                    # Get the smell_content (diff) and apply it
                    smell_content = case_data.get("smell_content", "")
                    if smell_content:
                        # Write diff to temporary file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding='utf-8') as tmp:
                            tmp.write(smell_content)
                            tmp_path = tmp.name

                        try:
                            # Ensure repository is in clean state before applying patch
                            print(f"[MockAgent] Resetting repository to clean state...")
                            reset_result = subprocess.run(
                                ['git', 'reset', '--hard'],
                                cwd=repo_path,
                                capture_output=True,
                                text=True
                            )
                            if reset_result.returncode != 0:
                                print(f"[MockAgent] Warning: git reset failed: {reset_result.stderr}")

                            # Clean untracked files
                            clean_result = subprocess.run(
                                ['git', 'clean', '-fd'],
                                cwd=repo_path,
                                capture_output=True,
                                text=True
                            )
                            if clean_result.returncode != 0:
                                print(f"[MockAgent] Warning: git clean failed: {clean_result.stderr}")

                            # Apply the diff using git apply
                            result = subprocess.run(
                                ['git', 'apply', tmp_path],
                                cwd=repo_path,
                                capture_output=True,
                                text=True
                            )
                            if result.returncode == 0:
                                print(f"[MockAgent] Successfully applied real case diff")
                                return (target_file, target_class, target_method)
                            else:
                                print(f"[MockAgent] Warning: Failed to apply diff: {result.stderr}")
                        finally:
                            os.unlink(tmp_path)
            except Exception as e:
                print(f"[MockAgent] Warning: Failed to load/apply template: {e}")
                import traceback
                traceback.print_exc()

        # Fallback: return None to indicate failure
        print(f"[MockAgent] Could not apply real case diff")
        return (None, None, None)

    def _detect_language(self, cwd):
        """Detect project language from file extensions."""
        if not cwd or not os.path.exists(cwd):
            return "python"

        # Count files by extension
        lang_files = {"python": 0, "java": 0, "go": 0}

        for root, dirs, files in os.walk(cwd):
            # Skip hidden and test directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]

            for file in files:
                if file.endswith('.py'):
                    lang_files["python"] += 1
                elif file.endswith('.java'):
                    lang_files["java"] += 1
                elif file.endswith('.go'):
                    lang_files["go"] += 1

        # Return language with most files
        return max(lang_files, key=lang_files.get)

    def _get_fallback_injection_response(self, cwd=None):
        """Fallback response when example cases are not available."""
        response = f"""I'll inject a code smell into the codebase.

```json
{{
  "smell_type": "Long Parameter List",
  "hint_targeted": "Mock smell injection",
  "hint_guided": "Check for methods with many parameters",
  "hint_open": "Examine the code for parameter issues",
  "smell_function": ["src/mock_file.py", "MockClass", "mock_method"],
  "test_functions": [["src/mock_file.py", "MockClass", "mock_method"]]
}}
```

The smell has been successfully injected."""
        return response

    def _get_fix_response(self):
        """Generate a response for fix prompts."""
        response = """I've analyzed the test failures. The mock smell injection should not affect existing tests.

The injected code is designed to be test-safe.

The smell remains intact for evaluation purposes."""
        return response

    def _get_analysis_response(self):
        """Generate a response for smell analysis prompts with proper rubrics format."""
        response = """<analysis>
This code demonstrates a clear code smell pattern that should be refactored. The implementation shows problematic design choices that reduce maintainability and readability. The specific issues include overly complex parameter handling, tight coupling between components, and violation of single responsibility principle.

Key concerns:
1. The parameter list has grown excessively long, making the method difficult to use and understand
2. The function signature is fragile and hard to extend without breaking existing callers
3. Related configuration options are not grouped logically
4. The implementation mixes concerns that should be separated

Refactoring recommendations:
- Introduce a configuration object or builder pattern to encapsulate related parameters
- Consider using keyword-only arguments in Python or similar patterns in other languages
- Group related parameters into logical data structures
- Apply the parameter object pattern to reduce coupling
</analysis>

<rubric>
<name>Core Smell Resolution</name>
<description>Correctly identifies and addresses the primary code smell pattern</description>
<excellent>9-10: Precisely identifies the smell type, provides specific code examples demonstrating the issue, and explains the underlying design principle violation with clear reasoning</excellent>
<good>7-8: Identifies the smell type correctly and points to relevant code sections that demonstrate the problem</good>
<acceptable>5-6: Recognizes some aspects of the smell but misses key manifestations or provides limited code examples</acceptable>
<below_average>3-4: Identifies a code issue but mischaracterizes the smell type or focuses on superficial symptoms</below_average>
<poor>0-2: Fails to identify the core smell or provides incorrect analysis</poor>
</rubric>

<rubric>
<name>Refactoring Strategy Quality</name>
<description>Proposes effective and practical refactoring approaches to eliminate the smell</description>
<excellent>9-10: Offers multiple concrete refactoring approaches with trade-offs, includes implementation guidance or pseudo-code, and explains how each approach addresses the root cause</excellent>
<good>7-8: Suggests at least one valid refactoring approach with clear explanation of how it resolves the smell</good>
<acceptable>5-6: Mentions refactoring strategies but lacks detail or doesn't fully address the smell's root cause</acceptable>
<below_average>3-4: Suggests superficial changes that don't address the underlying design issue</below_average>
<poor>0-2: Proposes ineffective or incorrect refactoring strategies</poor>
</rubric>"""
        return response

    def _tackle_output_to_response(self, model, output_text) -> AgentResponse:
        """Convert mock output to AgentResponse."""
        # Mock trajectory with some typical agent actions
        trajectory = [
            {"type": "read", "file": "src/utils/processor.py", "success": True},
            {"type": "edit", "file": "src/utils/processor.py", "success": True},
            {"type": "read", "file": "tests/test_processor.py", "success": True},
        ]

        # Mock reasonable token usage
        prompt_tokens = 1500
        completion_tokens = 800
        cache_read_tokens = 500
        cache_creation_tokens = 0

        return AgentResponse(
            content=output_text,
            model=model or self.model,
            raw_response=output_text,
            trajectory=trajectory,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cache_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            duration=2.5,  # Mock 2.5 seconds
            num_turns=3,
            tool_calls=3,
            tool_call_success=3,
            api_duration=2.0,
            total_cost_usd=0.05  # Mock cost
        )
