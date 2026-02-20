# Stats-Guided Black-Box Adversarial Evasion of Windows Malware Detectors

This repository contains the full implementation of the attack pipeline described in our paper:

> **Stats-Guided Black-Box Adversarial Evasion of Windows Malware Detectors**  
> Ayoub Mendoubi, Abdeslam El Fergougui

---

## Overview

We propose a two-stage black-box evasion attack against MalConv:

1. **Stage 1 — Pool Construction**: score a benign PE corpus through the target detector and retain only the *K* files with the lowest malicious probability.
2. **Stage 2 — Genetic Optimisation**: evolve adversarial byte payloads drawn from this filtered pool and append them to the PE overlay.

Compared with GAMMA under identical hyperparameters and a strict 20 KB payload budget:

| Method     | ASR    | Avg Queries | Avg Payload |
|------------|--------|-------------|-------------|
| GAMMA      | 77.7%  | 109         | 13.8 KB     |
| Stats-K25  | 92.3%  | 53          | 11.1 KB     |

---

## Repository Structure

```
.
├── detector/
│   └── train_malconv.py       # MalConv training script
├── attack/
│   ├── pools.py               # ContentPool, GammaPool, StatsGuidedPool
│   ├── operators.py           # GA operators: crossover, mutate, select
│   ├── ga.py                  # Core genetic algorithm loop
│   └── run_ablation.py        # Main entry point: K-ablation experiment
├── configs/
│   └── default.py             # All hyperparameters in one place
├── scripts/
│   ├── train.sh               # One-command training
│   └── attack.sh              # One-command attack
├── results/                   # Output directory (CSV, logs, adversarial files)
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.9+
- TensorFlow 2.20.0
- NumPy
- tqdm
- scipy
- pefile

---

## Usage

### 1. Train MalConv

```bash
python detector/train_malconv.py
```

Edit paths at the top of the file or pass them via `configs/default.py`.  
The trained model is saved as `malconv_robust_final.h5`.

### 2. Run the K-Ablation Attack

```bash
python attack/run_ablation.py
```

Results are written to `results/ablation_results.csv` and  
`results/ablation_summary.txt`.

### 3. One-Command Scripts

```bash
bash scripts/train.sh
bash scripts/attack.sh
```

---

## Configuration

All hyperparameters are centralised in `configs/default.py`:

```python
MAX_PAYLOAD_BYTES  = 20_000   # 20 KB payload cap
MAX_QUERIES        = 300      # query budget per sample
N_BENIGN_FILES     = 1000     # benign corpus size
TOP_K_VALUES       = [10, 25, 50, 100, 250]
```

---

## Reproducibility

- All random seeds are derived deterministically from file names via MD5.
- The benign pool is scored exactly once and shared across all K variants.
- Malware sample order is fixed with `SHUFFLE_SEED = 42`.

---

## Citation

```bibtex
@inproceedings{mendoubi2026statsguided,
  title     = {Stats-Guided Black-Box Adversarial Evasion of Windows Malware Detectors},
  author    = {Mendoubi, Ayoub and El Fergougui, Abdeslam},
  booktitle = {[Conference Name]},
  year      = {2026}
}
```

---

## Ethical Considerations

All experiments were conducted in an isolated offline environment.  
No adversarial samples were submitted to public services.  
Malware was obtained from established research repositories under their terms of use.
