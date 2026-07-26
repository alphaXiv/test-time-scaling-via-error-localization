#!/usr/bin/env bash
set -euo pipefail
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
torchrun --standalone --nproc_per_node=4 run_reproduction.py

