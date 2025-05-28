#!/bin/bash

make html

# open docs/_build/html/index.html in browser

python3 -m http.server 8000
