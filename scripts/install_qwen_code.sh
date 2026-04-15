#!/bin/bash
# Install Qwen Code CLI
set -e

if command -v qwen &> /dev/null; then
    echo "Qwen Code CLI already installed: $(qwen --version 2>/dev/null || echo 'unknown version')"
    exit 0
fi

echo "Installing Qwen Code CLI..."

# Ensure Node.js is available
if ! command -v node &> /dev/null; then
    echo "  Installing Node.js via conda..."
    conda install -y -c conda-forge nodejs
fi

npm install -g qwen-coder-cli

echo "Qwen Code CLI installed: $(qwen --version 2>/dev/null || echo 'done')"
