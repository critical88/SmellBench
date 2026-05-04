"""
Claude CLI Helpers
==================
Shared utilities for calling the Claude CLI, the Anthropic API, and parsing responses.
Extracted to avoid circular imports between smell_benchmark and find_candidates.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CLAUDE_CMD_TEMPLATE = "claude -p --permission-mode acceptEdits --verbose --output-format stream-json"


def _print_event_info(event: dict):
    """Print condensed human-readable event information in real-time."""
    event_type = event.get("type", "unknown")
    message = event.get("message", {})

    if event_type in ("assistant", "message"):
        content_array = message.get("content", event.get("content", []))
        if isinstance(content_array, list):
            for item in content_array:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        display = text[:200] + "..." if len(text) > 200 else text
                        print(f"    Assistant: {display}", flush=True)
                elif item_type == "tool_use":
                    tool_name = item.get("name", "")
                    tool_input = item.get("input", {})
                    print(f"    Tool: {tool_name}", flush=True)
                    if isinstance(tool_input, dict):
                        for key, value in list(tool_input.items())[:3]:
                            if key in ("content", "new_string", "old_string"):
                                print(f"      {key}: ({len(str(value))} chars)", flush=True)
                            else:
                                v = str(value)
                                print(f"      {key}: {v[:80]}{'...' if len(v) > 80 else ''}", flush=True)

    elif event_type == "tool_result":
        content_array = event.get("content", message.get("content", []))
        if isinstance(content_array, list):
            for item in content_array:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        lines = text.split("\n")
                        print(f"    Tool Result: ({len(lines)} lines)", flush=True)
                        break

    elif event_type == "result":
        result_text = event.get("result", "")
        usage = event.get("usage", {})
        print(f"    Final Result: ({len(result_text)} chars)", flush=True)
        if usage:
            print(
                f"      Tokens: in={usage.get('input_tokens', 0)}, "
                f"out={usage.get('output_tokens', 0)}, "
                f"cost=${event.get('total_cost_usd', 0):.4f}",
                flush=True,
            )

    elif event_type == "error":
        print(f"    Error: {event.get('message', message.get('message', ''))}", flush=True)


def extract_usage(envelope: dict) -> Dict:
    """Extract token usage and cost from a result envelope dict."""
    usage = envelope.get("usage", {})
    return {
        "input_tokens": (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        ),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "total_cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_ms": envelope.get("duration_ms", 0),
    }


def call_claude_cli(
    prompt: str, cwd: str, timeout: int = 1200, model: str = ""
) -> Tuple[str, List[Dict], Dict]:
    """Call claude CLI with streaming output, trajectory capture, and usage tracking.

    Returns:
        (result_text, trajectory, usage_dict)
    """
    command = shlex.split(CLAUDE_CMD_TEMPLATE)
    if model:
        command.extend(["--model", model])
    agent_cmd = shutil.which(command[0])
    if agent_cmd is None:
        raise RuntimeError("claude CLI not found in PATH")
    command[0] = agent_cmd

    trajectory: List[Dict] = []
    result_envelope: Optional[Dict] = None
    accumulated_usage: Dict[str, Any] = {}

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Send prompt and close stdin
    process.stdin.write(prompt)
    process.stdin.close()

    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            process.kill()
            raise RuntimeError(f"claude CLI timed out after {timeout}s")

        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue

        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
            trajectory.append(event)
            _print_event_info(event)

            # Accumulate usage from all events that have it
            if "usage" in event:
                event_usage = extract_usage(event)
                for key, value in event_usage.items():
                    if key in accumulated_usage:
                        accumulated_usage[key] += value
                    else:
                        accumulated_usage[key] = value

            if event.get("type") == "result":
                result_envelope = event
        except json.JSONDecodeError:
            continue

    returncode = process.wait()
    stderr = process.stderr.read()

    if returncode != 0:
        raise RuntimeError(
            f"claude CLI failed with code {returncode}: {stderr[:500]}"
        )

    result_text = result_envelope.get("result", "") if result_envelope else ""

    # Use accumulated usage from all events
    usage = accumulated_usage if accumulated_usage else {}

    return result_text, trajectory, usage


def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract the JSON object from the agent's response text.

    Tries several strategies:
    1. Find JSON in ```json ... ``` fenced block
    2. Find the last { ... } block in the text
    """
    if not response_text:
        return None

    # Strategy 1: fenced code block
    pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass

    # Strategy 2: last top-level JSON object
    last_brace = response_text.rfind("{")
    while last_brace != -1:
        candidate = response_text[last_brace:]
        depth = 0
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[: i + 1])
                    except json.JSONDecodeError:
                        break
        last_brace = response_text.rfind("{", 0, last_brace)

    return None


# ---------------------------------------------------------------------------
# Anthropic API helpers
# ---------------------------------------------------------------------------

def _load_env_file() -> None:
    """Load .env file from the same directory as this script, without overriding existing env vars."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in os.environ:
            os.environ[key] = value


def _parse_model_spec(model: str) -> Tuple[str, str]:
    """Parse a 'provider/model-name' string into (provider, model_name).

    If no '/' is present, defaults to 'anthropic' provider for backward compatibility.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider.lower(), model_name
    return "anthropic", model


def _call_mock(
    prompt: str,
    model: str,
) -> Dict[str, Any]:
    """Return mock responses for testing without API calls."""
    # Detect if this is a smell analysis prompt
    if "smell analysis" in prompt.lower() or "<analysis>" in prompt.lower():
        # Return mock smell analysis with proper rubric format (all 5 scoring levels required)
        raw_text = """<analysis>
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
    else:
        # Generic mock response
        raw_text = """The analysis has been completed successfully. This is a mock response for testing purposes."""

    usage = {
        "input_tokens": 1200,
        "output_tokens": 600,
        "duration_ms": 2000,
    }

    parsed = extract_json_from_response(raw_text)
    return {"parsed": parsed, "raw": raw_text, "usage": usage}


def _call_anthropic(
    prompt: str,
    model: str,
    max_tokens: int,
    base_url: Optional[str],
) -> Dict[str, Any]:
    """Call the Anthropic API."""
    import anthropic

    kwargs: Dict[str, Any] = {}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    effective_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    client = anthropic.Anthropic(**kwargs)
    start_ms = time.time() * 1000
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    duration_ms = int(time.time() * 1000 - start_ms)
    raw_text = message.content[0].text
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "duration_ms": duration_ms,
    }
    parsed = extract_json_from_response(raw_text)
    return {"parsed": parsed, "raw": raw_text, "usage": usage}


def _call_openai(
    prompt: str,
    model: str,
    max_tokens: int,
    base_url: Optional[str],
) -> Dict[str, Any]:
    """Call an OpenAI-compatible API."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    effective_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    client = openai.OpenAI(**kwargs)
    start_ms = time.time() * 1000
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    duration_ms = int(time.time() * 1000 - start_ms)
    raw_text = response.choices[0].message.content or ""
    usage_data = response.usage
    usage = {
        "input_tokens": usage_data.prompt_tokens if usage_data else 0,
        "output_tokens": usage_data.completion_tokens if usage_data else 0,
        "duration_ms": duration_ms,
    }
    parsed = extract_json_from_response(raw_text)
    return {"parsed": parsed, "raw": raw_text, "usage": usage}


def call_llm(
    prompt: str,
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    max_tokens: int = 8192,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Call an LLM API. Model format: 'provider/model-name'.

    Supported providers: anthropic, openai, mock.
    If no provider prefix, defaults to anthropic for backward compatibility.
    Use "mock" or "test" as model name to get mock responses without API calls.

    Returns:
        Dict with keys: "parsed" (the parsed JSON result or None),
        "raw" (raw response text), "usage" (token usage dict).
    """
    _load_env_file()

    # Check for mock mode
    if model.lower() in ("mock", "test", "mock-agent-v1"):
        return _call_mock(prompt, model)

    provider, model_name = _parse_model_spec(model)

    if provider == "anthropic":
        return _call_anthropic(prompt, model_name, max_tokens, base_url)
    elif provider == "openai":
        return _call_openai(prompt, model_name, max_tokens, base_url)
    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Use 'anthropic/...' or 'openai/...'")
