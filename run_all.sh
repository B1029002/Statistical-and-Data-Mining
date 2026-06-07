#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- reproduce the entire CE5033 Final Project analysis end-to-end.
#
# Usage:
#   bash run_all.sh
#
# It runs the four numbered stages in order. Each stage writes its outputs to
# data/, figures/ and results/. Stages 2-4 depend only on the cleaned files
# produced by stage 1, so the pipeline is fully reproducible from the raw CSV.
# =============================================================================
set -euo pipefail

# --- pick the Python interpreter --------------------------------------------
# Prefer the project conda env; fall back to whatever `python` is on PATH.
PY="/home/u5534225/miniconda3/envs/data_mining/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python)"; fi
echo "Using interpreter: $PY"
"$PY" --version

cd "$(dirname "$0")/src"

echo
echo ">>> STEP 1/4  Data preprocessing"
"$PY" 01_preprocessing.py

echo
echo ">>> STEP 2/4  Exploratory data analysis"
"$PY" 02_eda.py

echo
echo ">>> STEP 3/4  Statistical methods"
"$PY" 03_statistics.py

echo
echo ">>> STEP 4/4  Data mining"
"$PY" 04_datamining.py

echo
echo "All stages finished. See data/, figures/ and results/."
