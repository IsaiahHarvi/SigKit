#!/bin/bash
set -euo pipefail

rm -rf docs/api

sphinx-apidoc \
  --force \
  --module-first \
  --output-dir docs/api \
  src/sigkit

cd docs
make html

# open docs/_build/html/index.html in browser

python3 -m http.server 8000
