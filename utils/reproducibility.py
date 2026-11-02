"""Reproducibility helpers: one seed controls every randomness source."""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> np.random.Generator:
    """Seed ``random`` and NumPy; returns a fresh default_rng(seed)."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    return np.random.default_rng(int(seed))
