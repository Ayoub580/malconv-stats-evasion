#!/bin/bash
# scripts/attack.sh — run the K-ablation experiment
set -e
cd "$(dirname "$0")/.."
python attack/run_ablation.py
