#!/bin/bash
# scripts/train.sh — train MalConv
set -e
cd "$(dirname "$0")/.."
python detector/train_malconv.py
