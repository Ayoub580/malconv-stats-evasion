# attack/ga.py
"""
Core genetic algorithm: Individual dataclass, population evaluation,
and the main optimisation loop.
"""

import copy
import hashlib
import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Individual ────────────────────────────────────────────────

@dataclass
class Individual:
    genes:    np.ndarray
    fitness:  float = float("inf")
    mal_prob: float = float("inf")

    @property
    def size(self) -> int:
        return len(self.genes)


# ── Population evaluation ─────────────────────────────────────

def evaluate_population(malware_bytes: bytes, individuals: List[Individual],
                         detector, lambda_size: float, max_size: int) -> int:
    """
    Score a list of individuals in batches through the detector.
    Updates fitness and mal_prob in place. Returns number of queries used.
    """
    if not individuals:
        return 0
    raws   = [malware_bytes + ind.genes.tobytes() for ind in individuals]
    scores = detector.score_batch(raws)
    for ind, score in zip(individuals, scores):
        ind.mal_prob = float(score)
        ind.fitness  = float(score) + lambda_size * (ind.size / max(max_size, 1))
    return len(individuals)


# ── Seed helpers ──────────────────────────────────────────────

def _stable_pool_seed(pool_name: str) -> int:
    """
    Derive a deterministic integer seed from a pool name using MD5.
    Python's built-in hash() is randomised per process by PYTHONHASHSEED
    and must not be used for reproducible seeding.
    """
    return int(hashlib.md5(pool_name.encode()).hexdigest()[:8], 16)


# ── Core loop ─────────────────────────────────────────────────

def _run_single_ga(malware_bytes: bytes, pool, detector, cfg) -> tuple:
    """
    Run one (malware sample, pool) pair through the full GA.
    Returns (best_individual, queries_used).
    Returns (None, 0) if the file is too large to attack.
    """
    from attack.operators import tournament_select, crossover, mutate

    eff_max = min(cfg.max_payload_bytes, cfg.padding_size - len(malware_bytes))
    if eff_max < cfg.min_payload_bytes:
        return None, 0

    n_elite    = max(1, int(cfg.population_size * cfg.elite_frac))
    queries    = 0
    stagnation = 0

    pop     = [pool.make_individual(eff_max) for _ in range(cfg.population_size)]
    queries += evaluate_population(malware_bytes, pop, detector,
                                   cfg.lambda_size, eff_max)
    pop.sort(key=lambda i: i.fitness)
    best = copy.deepcopy(pop[0])

    while queries < cfg.max_queries:
        remaining   = cfg.max_queries - queries
        n_offspring = cfg.population_size - n_elite
        if remaining < n_offspring:
            break

        new_pop   = copy.deepcopy(pop[:n_elite])
        offspring = []
        while len(offspring) < n_offspring:
            p1 = tournament_select(pop, cfg.tournament_size)
            p2 = tournament_select(pop, cfg.tournament_size)
            c1, c2 = crossover(p1, p2, cfg.crossover_rate, eff_max)
            offspring.extend([
                mutate(c1, cfg.mutation_rate, pool, eff_max),
                mutate(c2, cfg.mutation_rate, pool, eff_max),
            ])
        offspring = offspring[:n_offspring]
        queries  += evaluate_population(malware_bytes, offspring, detector,
                                        cfg.lambda_size, eff_max)

        pop = new_pop + offspring
        pop.sort(key=lambda i: i.fitness)

        if pop[0].fitness < best.fitness:
            best, stagnation = copy.deepcopy(pop[0]), 0
        else:
            stagnation += 1

        # Stagnation reset: replace bottom of population with fresh individuals
        if stagnation >= cfg.local_min_patience:
            remaining = cfg.max_queries - queries
            n_fresh   = min(cfg.population_size - n_elite, remaining)
            if n_fresh <= 0:
                break
            fresh   = [pool.make_individual(eff_max) for _ in range(n_fresh)]
            queries += evaluate_population(malware_bytes, fresh, detector,
                                           cfg.lambda_size, eff_max)
            pop     = pop[:n_elite] + fresh
            pop.sort(key=lambda i: i.fitness)
            stagnation = 0

        if best.mal_prob < cfg.evasion_threshold:
            break

    return best, queries


def run_ga(malware_bytes: bytes, pool, detector, cfg,
           sample_id: str = "", sample_seed: Optional[int] = None) -> dict:
    """
    Public entry point for one (sample, pool) evaluation.
    Seeds random state deterministically from sample_seed XOR pool name hash,
    ensuring full reproducibility across runs.
    """
    if sample_seed is not None:
        method_seed = sample_seed ^ _stable_pool_seed(pool.name)
        random.seed(method_seed)
        np.random.seed(method_seed % (2 ** 32))

    t_start       = time.time()
    best, queries = _run_single_ga(malware_bytes, pool, detector, cfg)

    if best is None:
        return dict(
            attack=pool.name, sample_id=sample_id,
            success=False, final_score=1.0,
            queries_used=queries, payload_bytes=0,
            overhead_pct=0.0,
            time_s=round(time.time() - t_start, 2),
            _payload=None,
        )

    return dict(
        attack        = pool.name,
        sample_id     = sample_id,
        success       = bool(best.mal_prob < cfg.evasion_threshold),
        final_score   = round(best.mal_prob, 6),
        queries_used  = queries,
        payload_bytes = best.size,
        overhead_pct  = round(best.size / max(1, len(malware_bytes)) * 100, 1),
        time_s        = round(time.time() - t_start, 2),
        _payload      = best.genes,
    )
