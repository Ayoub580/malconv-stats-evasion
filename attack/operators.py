# attack/operators.py
"""
Genetic algorithm operators: tournament selection, crossover, mutation.
All operators are stateless functions — they depend only on their inputs
and the global random state, which is seeded deterministically per sample.
"""

import copy
import random
from typing import Tuple

import numpy as np

from attack.ga import Individual


def tournament_select(population: list, k: int) -> Individual:
    """Return the fittest individual from a random tournament of size k."""
    return min(random.sample(population, k), key=lambda ind: ind.fitness)


def crossover(a: Individual, b: Individual,
              rate: float, max_size: int) -> Tuple[Individual, Individual]:
    """
    Single-point crossover. Returns two children.
    If the crossover rate is not triggered, copies of the parents are returned.
    """
    if random.random() > rate:
        return copy.deepcopy(a), copy.deepcopy(b)

    max_pt = min(len(a.genes), len(b.genes))
    if max_pt < 2:
        return copy.deepcopy(a), copy.deepcopy(b)

    pt = random.randint(1, max_pt - 1)
    c1 = np.concatenate([a.genes[:pt], b.genes[pt:]])[:max_size]
    c2 = np.concatenate([b.genes[:pt], a.genes[pt:]])[:max_size]
    return Individual(c1), Individual(c2)


def mutate(ind: Individual, rate: float, pool,
           max_size: int) -> Individual:
    """
    Two-phase mutation applied with probability `rate`:
      1. Content mutation — replace a random sub-sequence with bytes
         drawn from the pool.
      2. Length mutation — randomly truncate or extend the byte array,
         always staying within [pool.min_size, max_size].
    """
    if random.random() > rate:
        return copy.deepcopy(ind)

    genes = ind.genes.copy()

    # Phase 1: content substitution
    if len(genes) > 1 and random.random() < 0.5:
        s       = random.randint(0, len(genes) - 1)
        e       = random.randint(s + 1, len(genes))
        genes[s:e] = pool.sample_bytes(e - s)

    # Phase 2: length adjustment
    if random.random() < 0.5:
        if random.random() < 0.5 and len(genes) > pool.min_size:
            genes = genes[:random.randint(pool.min_size, len(genes) - 1)]
        else:
            room = max_size - len(genes)
            if room > 0:
                extra = pool.sample_bytes(random.randint(1, room))
                genes = np.concatenate([genes, extra])

    return Individual(genes)
