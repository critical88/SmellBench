"""
Claude CLI Helpers
==================
Shared utilities for calling the Claude CLI and parsing responses.
Extracted to avoid circular imports between smell_benchmark and find_candidates.
"""

import json
import re
import shlex
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple


CLAUDE_CMD_TEMPLATE = "claude -p --verbose --output-format stream-json"


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
    prompt: str, cwd: str, timeout: int = 1200
) -> Tuple[str, List[Dict], Dict]:
    """Call claude CLI with streaming output, trajectory capture, and usage tracking.

    Returns:
        (result_text, trajectory, usage_dict)
    """
    command = shlex.split(CLAUDE_CMD_TEMPLATE)
    agent_cmd = shutil.which(command[0])
    if agent_cmd is None:
        raise RuntimeError("claude CLI not found in PATH")
    command[0] = agent_cmd

    trajectory: List[Dict] = []
    result_envelope: Optional[Dict] = None

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
    usage = extract_usage(result_envelope) if result_envelope else {}

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
