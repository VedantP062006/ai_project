"""Unit tests for feature extraction and zero-leakage verification."""

import pytest
import numpy as np
from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint
from features.extractor import FeatureExtractor, NUM_FEATURES
from generators.sudoku import generate_sudoku
from generators.graph_coloring import generate_graph_coloring


def test_feature_extractor_shape_and_mask():
    inst = generate_sudoku(seed=42, difficulty="easy")
    extractor = FeatureExtractor()
    features, candidate_mask = extractor.extract(inst.csp, depth=0)

    assert features.shape == (81, NUM_FEATURES)
    assert candidate_mask.shape == (81,)
    assert np.all(np.isfinite(features))

    # Assigned cells must have candidate_mask = False
    for v in range(81):
        if inst.csp.domain_size(v) == 1:
            assert not candidate_mask[v]
        else:
            assert candidate_mask[v]


def test_feature_extractor_no_ground_truth_leakage():
    """
    CRITICAL EVALUATION RULE:
    Assert that features depend solely on observable constraint/domain state,
    and are completely invariant to any external ground-truth solution metadata.
    """
    inst1 = generate_graph_coloring(num_vertices=20, num_colors=3, avg_degree=4.0, planted=True, seed=10)
    inst2 = inst1.clone()

    # Modify metadata and planted solution on inst2
    inst2.metadata["planted_solution"] = {v: 0 for v in range(20)}
    inst2.metadata["clues"] = 9999

    extractor = FeatureExtractor()
    feat1, mask1 = extractor.extract(inst1.csp, depth=0)
    feat2, mask2 = extractor.extract(inst2.csp, depth=0)

    # Features must be bit-for-bit identical regardless of metadata or external solution!
    np.testing.assert_array_equal(feat1, feat2)
    np.testing.assert_array_equal(mask1, mask2)
