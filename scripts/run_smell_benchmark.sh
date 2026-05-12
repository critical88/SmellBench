#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if project name is provided
if [ -z "$1" ]; then
    echo "Usage: bash run_smell_benchmark.sh <project-name> --agent <agent_type> [--force] [--model MODEL] [--base-url URL]"
    echo "Supported agents: claude_code, qwen_code, openhands, codex, mock, test"
    exit 1
fi

PROJECT_NAME=$1
shift

# Parse optional arguments
AGENT_TYPE="claude_code"
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --agent)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --force)
            EXTRA_ARGS="$EXTRA_ARGS --force"
            shift
            ;;
        --model)
            EXTRA_ARGS="$EXTRA_ARGS --model $2"
            shift 2
            ;;
        --base-url)
            EXTRA_ARGS="$EXTRA_ARGS --base-url $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "======================================="
echo "Starting smell case generation for: ${PROJECT_NAME}"
echo "Agent: ${AGENT_TYPE}"
echo "======================================="

# ---------------------------------------------------------------------------
# Step 1: Ensure AST analysis is done (prerequisite for function-test mapping)
# ---------------------------------------------------------------------------
# Skip AST analysis - smell_benchmark.py handles it automatically when needed
echo "Step 1: AST-based Analysis (skipped, handled by smell_benchmark.py)"

# ---------------------------------------------------------------------------
# Step 2: Install the code agent CLI via its dedicated install script
# ---------------------------------------------------------------------------
# Mock agent doesn't need installation - skip for mock and test agents
if [ "${AGENT_TYPE}" = "mock" ] || [ "${AGENT_TYPE}" = "test" ]; then
    echo "Step 2: Installing agent (skipped for ${AGENT_TYPE})"
else
    INSTALL_SCRIPT="scripts/install_${AGENT_TYPE}.sh"
    if [ ! -f "$INSTALL_SCRIPT" ]; then
        echo "ERROR: Install script not found: ${INSTALL_SCRIPT}"
        echo "Supported agents: claude_code, qwen_code, openhands, codex, mock, test"
        exit 1
    fi

    echo "Step 2: Installing agent (${INSTALL_SCRIPT})"
    bash "${INSTALL_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# Step 3: Run smell benchmark pipeline
# ---------------------------------------------------------------------------
echo "Step 3: Running smell benchmark pipeline (agent=${AGENT_TYPE})"
python -u smell_benchmark.py \
    --project-name "${PROJECT_NAME}" \
    --agent "${AGENT_TYPE}" \
    ${EXTRA_ARGS}

echo "======================================="
echo "Smell case generation completed for: ${PROJECT_NAME}"
echo "======================================="
