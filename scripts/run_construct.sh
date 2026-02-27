#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if project name is provided
if [ -z "$1" ]; then
    echo "Usage: bash run_construct.sh <project-name> [force]"
    exit 1
fi

PROJECT_NAME=$1
FORCE_FLAG=""

# If second argument exists, enable --force
if [ -n "$2" ]; then
    FORCE_FLAG="--force"
fi

echo "======================================="
echo "Starting benchmark construction for: ${PROJECT_NAME}"
echo "======================================="

echo "Step 1: AST-based Analysis"
conda run -n testbed python -u ast_analyze.py --project-name "${PROJECT_NAME}"

echo "Step 2: Smell Injection"
conda run -n testbed python -u smell_injection.py \
    --project-name "${PROJECT_NAME}" \
    ${FORCE_FLAG}

echo "======================================="
echo "Benchmark construction completed for: ${PROJECT_NAME}"
echo "======================================="