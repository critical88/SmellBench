#!/bin/bash
# Install OpenAI Codex CLI
set -e

if command -v codex &> /dev/null; then
    echo "Codex CLI already installed: $(codex --version 2>/dev/null || echo 'unknown version')"
    exit 0
fi

echo "Installing Codex CLI..."

# Ensure Node.js is available
if ! command -v node &> /dev/null; then
    echo "  Installing Node.js via conda..."
    conda install -y -c conda-forge nodejs
fi

npm install -g @openai/codex

echo "Codex CLI installed: $(codex --version 2>/dev/null || echo 'done')"
