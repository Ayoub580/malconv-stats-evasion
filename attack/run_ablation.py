# attack/run_ablation.py
"""
Main entry point for the K-ablation experiment.
Runs GAMMA and Stats-K{k} methods on up to MAX_SAMPLES malware files
and writes results to results/ablation_results.csv and ablation_summary.txt.
"""

import csv
import hashlib
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configs.default import (
    MODEL_PATH, BENIGN_DIR, MALWARE_DIR, OUTPUT_DIR,
    N_BENIGN_FILES, TOP_K_VALUES, MAX_SAMPLES, SHUFFLE_SEED,
    GAConfig,
)
from attack.detector import MalConvDetector
from attack.ga import run_ga
from attack.pools import (
    GammaPool, StatsGuidedPool,
    load_benign_files, score_benign_pool,
)

# ── Logging ───────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "run.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ── Pool diagnostic ───────────────────────────────────────────

def print_pool_diagnostic(pools: list, detector: MalConvDetector,
                           n: int = 20) -> None:
    print("\n" + "=" * 65)
    print("  POOL QUALITY DIAGNOSTIC (random sample of n files per pool)")
    print("=" * 65)
    for pool in pools:
        sample = random.sample(pool._arrays, min(n, len(pool._arrays)))
        scores = [detector.score_one(f) for f in sample]
        print(
            f"  {pool.name:<20}  "
            f"avg={np.mean(scores):.6f}  "
            f"min={min(scores):.6f}  "
            f"max={max(scores):.6f}  "
            f"n={len(sample)}"
        )
    print("=" * 65 + "\n")


# ── McNemar test ──────────────────────────────────────────────

def mcnemar_test(successes_a: list, successes_b: list) -> dict:
    from scipy.stats import chi2 as chi2_dist
    n01 = sum(1 for a, b in zip(successes_a, successes_b) if not a and b)
    n10 = sum(1 for a, b in zip(successes_a, successes_b) if a and not b)
    if n01 + n10 == 0:
        return dict(chi2=float("nan"), p_value=float("nan"),
                    n01=n01, n10=n10, note="No discordant pairs")
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value   = 1.0 - chi2_dist.cdf(chi2_stat, df=1)
    return dict(chi2=round(chi2_stat, 4), p_value=round(p_value, 5),
                n01=n01, n10=n10)


# ── Summary ───────────────────────────────────────────────────

def save_summary(records: list, top_k_values: list,
                 prefilter_elapsed: float) -> None:
    groups       = defaultdict(list)
    for r in records:
        groups[r["attack"]].append(r)

    attack_order = ["GAMMA"] + [f"Stats-K{k}" for k in top_k_values]
    lines        = ["=" * 65,
                    "  K-ABLATION SUMMARY",
                    "=" * 65]

    gamma_recs  = groups.get("GAMMA", [])
    gamma_asr   = None
    gamma_avg_q = None
    n_samples   = len(gamma_recs)
    n_k         = len(top_k_values)

    for attack in attack_order:
        recs = groups.get(attack, [])
        if not recs:
            continue
        n    = len(recs)
        succ = [r for r in recs if r["success"]]
        ns   = len(succ)
        asr  = ns / n * 100
        avg_q = sum(r["queries_used"] for r in recs) / n
        avg_t = sum(r["time_s"] for r in recs) / n

        if attack == "GAMMA":
            gamma_asr, gamma_avg_q = asr, avg_q

        prefilter_per_sample = (
            prefilter_elapsed / max(n_k * n_samples, 1)
            if attack != "GAMMA" else 0.0
        )

        lines += [
            f"\n  {attack}",
            f"    Evasion Rate        : {asr:.1f}%  ({ns}/{n})",
            f"    Avg Queries         : {avg_q:.0f}",
            f"    Avg Payload (all)   : "
            f"{sum(r['payload_bytes'] for r in recs)/n/1024:.1f} KB",
            f"    Avg Payload (succ.) : "
            f"{sum(r['payload_bytes'] for r in succ)/max(1,ns)/1024:.1f} KB",
            f"    Avg Overhead        : "
            f"{sum(r['overhead_pct'] for r in recs)/n:.1f}%",
            f"    Avg Det. Score      : "
            f"{sum(r['final_score'] for r in recs)/n:.4f}",
            f"    Avg GA Time (s)     : {avg_t:.2f}",
            f"    Prefilter/sample    : {prefilter_per_sample:.4f}s",
        ]

        if attack != "GAMMA" and gamma_asr is not None:
            mc  = mcnemar_test(
                [r["success"] for r in gamma_recs],
                [r["success"] for r in recs],
            )
            sig = "significant" if mc.get("p_value", 1) < 0.05 else "NOT significant"
            lines += [
                f"    vs GAMMA ASR Δ      : {asr - gamma_asr:+.1f}pp",
                f"    vs GAMMA Query Δ    : {avg_q - gamma_avg_q:+.0f}  "
                f"({(avg_q - gamma_avg_q)/max(gamma_avg_q,1)*100:+.1f}%)",
                f"    McNemar χ²          : {mc.get('chi2','n/a')}  "
                f"p={mc.get('p_value','n/a')}  [{sig}]",
                f"    Discordant pairs    : "
                f"GAMMA✓/Stats✗={mc.get('n10',0)}  "
                f"GAMMA✗/Stats✓={mc.get('n01',0)}",
            ]

    lines += [
        "\n" + "─" * 65,
        f"  Prefilter total : {prefilter_elapsed:.1f}s "
        f"over {N_BENIGN_FILES} files (scored once)",
        "=" * 65,
    ]

    summary = "\n".join(lines)
    print(summary)
    Path(OUTPUT_DIR, "ablation_summary.txt").write_text(summary)
    logger.info("Summary written to ablation_summary.txt")


# ── Main ──────────────────────────────────────────────────────

def main():
    tf.random.set_seed(42)

    cfg      = GAConfig()
    detector = MalConvDetector(MODEL_PATH)

    benign_arrays = load_benign_files(BENIGN_DIR, N_BENIGN_FILES)
    sorted_arrays, sorted_scores, prefilter_elapsed = score_benign_pool(
        benign_arrays, detector
    )

    gamma_pool  = GammaPool(benign_arrays)
    stats_pools = {
        k: StatsGuidedPool(sorted_arrays, sorted_scores, top_k=k)
        for k in TOP_K_VALUES
    }
    all_pools = [gamma_pool] + [stats_pools[k] for k in TOP_K_VALUES]

    # Finalise malware list before diagnostic to keep random state clean
    malware_paths = [p for p in Path(MALWARE_DIR).glob("*") if p.is_file()]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(malware_paths)
    malware_paths = malware_paths[:MAX_SAMPLES]
    logger.info(f"Processing {len(malware_paths)} malware samples")

    print_pool_diagnostic(all_pools, detector)

    for pool in all_pools:
        os.makedirs(
            os.path.join(OUTPUT_DIR, pool.name, "adversarial"), exist_ok=True
        )

    csv_path   = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    fieldnames = ["attack", "sample_id", "success", "final_score",
                  "queries_used", "payload_bytes", "overhead_pct", "time_s"]
    records    = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(
            tqdm(malware_paths, desc="Ablation", unit="sample")
        ):
            raw         = path.read_bytes()
            sid         = path.stem
            sample_seed = int(hashlib.md5(sid.encode()).hexdigest()[:8], 16)

            row_results = []
            for pool in all_pools:
                result  = run_ga(raw, pool, detector, cfg,
                                 sample_id=sid, sample_seed=sample_seed)
                payload = result.pop("_payload", None)

                if result["success"] and payload is not None:
                    adv_path = Path(
                        OUTPUT_DIR, pool.name, "adversarial", f"{sid}.exe"
                    )
                    adv_path.write_bytes(raw + payload.tobytes())

                records.append(result)
                writer.writerow(result)
                f.flush()
                row_results.append(result)

            parts = [f"[{i+1:>4}/{len(malware_paths)}] {sid[:24]:<24}"]
            for r in row_results:
                parts.append(
                    f"{r['attack']:<14} {'✓' if r['success'] else '✗'} "
                    f"s={r['final_score']:.3f} "
                    f"q={r['queries_used']:>4} "
                    f"sz={r['payload_bytes']//1024:>3}KB"
                )
            logger.info(" | ".join(parts))

    save_summary(records, TOP_K_VALUES, prefilter_elapsed)
    logger.info(f"Done. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
