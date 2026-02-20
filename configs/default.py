# configs/default.py
"""
Central configuration for the stats-guided evasion pipeline.
Edit this file to change any hyperparameter — all scripts import from here.
"""

from dataclasses import dataclass, field
from typing import List

# ── Paths ─────────────────────────────────────────────────────
MODEL_PATH  = "/home/ayoub/mal-research/malconv_robust_final.h5"
BENIGN_DIR  = "/home/ayoub/mal-research/datasets/balanced_dataset/test/benign"
MALWARE_DIR = "/home/ayoub/mal-research/datasets/balanced_dataset/test/malware"
OUTPUT_DIR  = "results"

TRAIN_BENIGN_PATH   = "/home/ayoub/mal-research/datasets/balanced_dataset/train/benign"
TRAIN_MALWARE_PATH  = "/home/ayoub/mal-research/datasets/balanced_dataset/train/malware"
TEST_BENIGN_PATH    = "/home/ayoub/mal-research/datasets/balanced_dataset/test/benign"
TEST_MALWARE_PATH   = "/home/ayoub/mal-research/datasets/balanced_dataset/test/malware"

# ── Detector training ─────────────────────────────────────────
MAX_LEN       = 1024 * 1024   # 1 MB — MalConv receptive field
BATCH_SIZE    = 4
EMBEDDING_DIM = 8

# ── Ablation settings ─────────────────────────────────────────
N_BENIGN_FILES = 1000
TOP_K_VALUES   = [10, 25, 50, 100, 250]
MAX_SAMPLES    = 300
SHUFFLE_SEED   = 42           # controls malware sample order only

# ── GA hyperparameters ────────────────────────────────────────
@dataclass
class GAConfig:
    population_size:    int   = 30
    max_queries:        int   = 300      # query budget per sample
    padding_size:       int   = 2 ** 20  # MalConv receptive field
    mutation_rate:      float = 0.4
    crossover_rate:     float = 0.7
    tournament_size:    int   = 5
    lambda_size:        float = 0.0      # size penalty (0 = disabled)
    local_min_patience: int   = 20       # generations before stagnation reset
    elite_frac:         float = 0.1      # fraction of elites preserved
    evasion_threshold:  float = 0.5      # MalConv decision boundary
    max_payload_bytes:  int   = 20_000   # 20 KB payload cap
    min_payload_bytes:  int   = 64
