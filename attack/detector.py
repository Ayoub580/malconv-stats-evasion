# attack/detector.py
"""
MalConv detector wrapper.
Handles batched inference with automatic GPU OOM recovery.
"""

import logging
from typing import List

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class MalConvDetector:
    def __init__(self, model_path: str, max_len: int = 2 ** 20,
                 batch_size: int = 4):
        logger.info(f"Loading MalConv from {model_path} ...")
        self.model      = tf.keras.models.load_model(model_path)
        self.max_len    = max_len
        self.batch_size = batch_size
        # Warm-up pass to initialise GPU memory
        self.model(np.zeros((1, max_len), dtype=np.uint8), training=False)
        logger.info(f"MalConv ready (batch_size={self.batch_size})")

    def _prep(self, raw: bytes) -> np.ndarray:
        arr = np.frombuffer(raw[:self.max_len], dtype=np.uint8)
        return np.pad(arr, (0, max(0, self.max_len - len(arr))))

    def score_one(self, raw: bytes) -> float:
        """Score a single file. Used during pool pre-filtering."""
        return float(
            self.model(self._prep(raw)[np.newaxis, :], training=False)[0][0]
        )

    def score_batch(self, raws: List[bytes]) -> np.ndarray:
        """
        Score a list of files in mini-batches.
        Automatically halves the batch size on GPU out-of-memory errors.
        """
        results, i = [], 0
        while i < len(raws):
            chunk = raws[i: i + self.batch_size]
            batch = np.stack([self._prep(r) for r in chunk])
            try:
                scores = self.model(batch, training=False).numpy().flatten()
                results.append(scores)
                i += self.batch_size
            except tf.errors.ResourceExhaustedError:
                self.batch_size = max(1, self.batch_size // 2)
                logger.warning(
                    f"GPU OOM — batch_size reduced to {self.batch_size}"
                )
        return np.concatenate(results)
