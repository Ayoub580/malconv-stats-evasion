# attack/pools.py
"""
Benign content pools used by the genetic algorithm.

GammaPool      — uniform random selection from all N files (GAMMA baseline).
StatsGuidedPool — top-K lowest-scoring files from a pre-scored ranked list.
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import List

import numpy as np

from attack.ga import Individual

logger = logging.getLogger(__name__)


# ── Base pool ─────────────────────────────────────────────────

class ContentPool:
    """Shared interface for all pool implementations."""

    def __init__(self, byte_arrays: List[bytes], name: str):
        self.name    = name
        self._arrays = [b for b in byte_arrays if len(b) >= 64]
        if not self._arrays:
            raise ValueError(f"[{name}] Pool is empty after 64-byte filter.")
        self.min_size = 64
        logger.info(f"[{name}] Pool ready — {len(self._arrays)} files")

    def sample_bytes(self, n: int) -> np.ndarray:
        """Draw n contiguous bytes from a randomly chosen file in the pool."""
        src = random.choice(self._arrays)
        if len(src) >= n:
            start = random.randint(0, len(src) - n)
            return np.frombuffer(src[start:start + n], dtype=np.uint8).copy()
        arr   = np.frombuffer(src, dtype=np.uint8)
        tiled = np.tile(arr, (n // len(arr)) + 1)
        return tiled[:n].copy()

    def make_individual(self, max_size: int) -> Individual:
        return Individual(
            genes=self.sample_bytes(random.randint(self.min_size, max_size))
        )


# ── GAMMA baseline pool ───────────────────────────────────────

class GammaPool(ContentPool):
    """
    Draws byte content uniformly at random from all N benign files.
    Replicates the content selection strategy of GAMMA
    (Demetrio et al., 2021).
    """

    def __init__(self, byte_arrays: List[bytes]):
        super().__init__(byte_arrays, name="GAMMA")


# ── Stats-guided pool ─────────────────────────────────────────

class StatsGuidedPool(ContentPool):
    """
    Restricts the pool to the top-K benign files ranked by ascending
    detector score (lowest malicious probability = most useful payload).

    The ranked list is computed once externally (see run_ablation.py)
    and passed in — all K variants share the same scoring pass with
    no redundant inference.
    """

    def __init__(self, ranked_arrays: List[bytes], ranked_scores: List[float],
                 top_k: int):
        top_data   = ranked_arrays[:top_k]
        top_scores = ranked_scores[:top_k]

        # Diversity check
        hashes       = [hashlib.md5(b).hexdigest() for b in top_data]
        unique_count = len(set(hashes))
        if unique_count < top_k // 2:
            logger.warning(
                f"[Stats-K{top_k}] Low diversity: "
                f"{unique_count}/{top_k} unique files by MD5."
            )

        logger.info(
            f"[Stats-K{top_k}] score range: "
            f"{top_scores[0]:.6f} – {top_scores[-1]:.6f}  "
            f"mean={sum(top_scores)/len(top_scores):.6f}  "
            f"unique={unique_count}/{top_k}"
        )
        super().__init__(top_data, name=f"Stats-K{top_k}")


# ── I/O helpers ───────────────────────────────────────────────

def load_benign_files(benign_dir: str, n_files: int,
                      max_file_bytes: int = 2 ** 20) -> List[bytes]:
    """Load up to n_files benign PE files in deterministic sort order."""
    paths  = sorted(p for p in Path(benign_dir).glob("*") if p.is_file())
    paths  = paths[:n_files]
    arrays = []
    for p in paths:
        data = p.read_bytes()
        if len(data) > max_file_bytes:
            data = data[:max_file_bytes]
        arrays.append(data)
    logger.info(f"Loaded {len(arrays)} benign files from {benign_dir}")
    return arrays


def score_benign_pool(arrays: List[bytes], detector) -> tuple:
    """
    Score all benign files once through the detector and return two
    parallel lists sorted by ascending score (most benign first).
    """
    import time
    from tqdm import tqdm

    logger.info(f"Scoring {len(arrays)} benign files (one-time pass) ...")
    t0     = time.time()
    scored = []
    for data in tqdm(arrays, desc="Pre-filtering benign pool", unit="file"):
        scored.append((data, detector.score_one(data)))
    scored.sort(key=lambda x: x[1])
    elapsed = time.time() - t0

    logger.info(
        f"Pre-filter done in {elapsed:.1f}s  "
        f"score range: {scored[0][1]:.6f} – {scored[-1][1]:.6f}"
    )
    return [b for b, _ in scored], [s for _, s in scored], elapsed
