#!/usr/bin/env bash

set -euo pipefail

# Install UV
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "Skipping UV install, found existing."
fi

# Setup container or user environment
if [ -n "$IN_SIGKIT_CONTAINER" ]; then
    git config --global --add safe.directory /workspaces/SigKit
    uv pip install --system -r requirements.txt
else
    uv pip install -r requirements.txt
    echo "The IN_SIGKIT_CONTAINER enviornment variable is not exported, proceeded with user install."
fi
