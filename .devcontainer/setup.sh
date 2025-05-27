#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

git config --global --add safe.directory /workspaces/SigKit
git config --global pull.rebase true

uv pip install --system -r requirements.txt

