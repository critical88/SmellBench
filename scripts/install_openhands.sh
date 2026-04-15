#!/bin/bash
# Install OpenHands CLI
set -e

if command -v openhands &> /dev/null; then
    echo "OpenHands already installed: $(openhands --version 2>/dev/null || echo 'unknown version')"
    exit 0
fi

echo "Installing OpenHands..."

pip install openhands-ai

echo "OpenHands installed: $(openhands --version 2>/dev/null || echo 'done')"
