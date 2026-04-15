#!/bin/bash
# Install Claude Code CLI
set -e

if command -v claude &> /dev/null; then
    echo "Claude Code CLI already installed: $(claude --version 2>/dev/null || echo 'unknown version')"
    exit 0
fi

echo "Installing Claude Code CLI..."

# Ensure Node.js is available
if ! command -v node &> /dev/null; then
    echo "  Installing Node.js via conda..."
    conda install -y -c conda-forge nodejs
fi

npm install -g @anthropic-ai/claude-code

echo "Claude Code CLI installed: $(claude --version 2>/dev/null || echo 'done')"
