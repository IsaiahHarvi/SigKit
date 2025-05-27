!#/bin/bash

uv pip compile requirements.in --output-file requirements.txt

uv add -r requirements.txt
