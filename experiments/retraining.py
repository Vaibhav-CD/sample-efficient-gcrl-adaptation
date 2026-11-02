"""Retraining baseline: the fair comparison for every adaptation claim.

Procedure: after the environment changes, discard all learned values and
train a *fresh* learner from scratch in the changed environment. Any
knowledge-reuse method (Stage 4+) must beat this baseline's post-change
interaction count at equal final success — otherwise it adds complexity for
no sample-efficiency gain. This module implements the procedure once so all
later experiments share it.
"""

from __future__ import annotations


def retrain_from_scratch(learner_factory, env, episodes: int, **train_kwargs):
    """Create a fresh learner via ``learner_factory()`` and train it.

    Returns ``(learner, history)``. ``learner_factory`` is a zero-argument
    callable so each call yields independent initial values.
    """
    learner = learner_factory()
    history = learner.train(env, episodes, **train_kwargs)
    return learner, history
