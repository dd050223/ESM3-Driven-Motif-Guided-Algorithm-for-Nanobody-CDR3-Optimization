#!/usr/bin/env bash
# Run ESM3 Nanobody CDR3 Optimization Pipeline

set -euo pipefail

echo "========================================"
echo "ESM3 Nanobody CDR3 Optimization"
echo "========================================"

# Determine Python executable
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
    echo "Installing dependencies..."
    $PYTHON -m pip install --upgrade pip
    $PYTHON -m pip install -r requirements.txt
fi

# Run the pipeline
echo "Running optimization pipeline..."
$PYTHON -m src.esm3_nanobody.cli run --config configs/default_config.json

echo ""
echo "========================================"
echo "Pipeline completed!"
echo "Results saved to outputs/"
echo "========================================"
