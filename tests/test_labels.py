"""Unit tests for oracle label generation and dataset trajectory collection."""

import pytest
import numpy as np
from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint
from labels.oracle import mrv_imitation_oracle, bounded_lookahead_oracle
from labels.dataset import collect_training_trajectories, SearchStateSample
from generators.sudoku import generate_sudoku
from generators.graph_coloring import generate_graph_coloring


def test_mrv_imitation_oracle():
    csp = CSP(num_variables=3, initial_domains=[values_to_bitmask([0, 1, 2]), values_to_bitmask([0, 1]), values_to_bitmask([0, 1, 2, 3])])
    candidates = [0, 1, 2]
    scores = mrv_imitation_oracle(csp, candidates)

    assert len(scores) == 3
    # Candidate 1 has smallest domain size (2) -> must receive top score
    best_idx = int(np.argmax(scores))
    assert candidates[best_idx] == 1


def test_bounded_lookahead_oracle():
    inst = generate_graph_coloring(num_vertices=15, num_colors=3, avg_degree=4.0, planted=True, seed=33)
    candidates = [0, 1, 2, 3]
    scores = bounded_lookahead_oracle(inst.csp, candidates, lookahead_depth=1)

    assert len(scores) == 4
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert np.max(scores) == 1.0


def test_collect_trajectories():
    instances = [generate_sudoku(seed=i, difficulty="easy") for i in range(2)]
    samples = collect_training_trajectories(
        instances=instances,
        oracle_type="mrv",
        max_states_per_instance=5,
        top_k_candidates=3,
        seed=42,
    )

    assert len(samples) > 0
    s = samples[0]
    assert isinstance(s, SearchStateSample)
    assert len(s.candidate_indices) <= 3
    assert s.candidate_features.shape[0] == len(s.candidate_indices)
    assert len(s.target_scores) == len(s.candidate_indices)
