"""Unit tests for LightGBM and PyTorch MLP models."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from models.lightgbm_model import LightGBMModel
from models.mlp_model import MLPModel
from generators.sudoku import generate_sudoku
from heuristics.variable_ordering import get_top_k_mrv_candidates


def test_lightgbm_fit_score_save_load():
    X = np.random.randn(50, 12).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)

    model = LightGBMModel(seed=42)
    model.fit(X, y)

    cand_feats = np.random.randn(5, 12).astype(np.float32)
    scores = model.score(cand_feats)
    assert scores.shape == (5,)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "lgbm.pkl"
        model.save(str(save_path))
        loaded = LightGBMModel.load(str(save_path))
        loaded_scores = loaded.score(cand_feats)
        np.testing.assert_allclose(scores, loaded_scores, rtol=1e-5)


def test_mlp_fit_score_save_load():
    X = np.random.randn(50, 12).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)

    model = MLPModel(seed=42)
    model.fit(X, y, epochs=5, batch_size=16)

    cand_feats = np.random.randn(5, 12).astype(np.float32)
    scores = model.score(cand_feats)
    assert scores.shape == (5,)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "mlp.pt"
        model.save(str(save_path))
        loaded = MLPModel.load(str(save_path))
        loaded_scores = loaded.score(cand_feats)
        np.testing.assert_allclose(scores, loaded_scores, rtol=1e-5)


def test_model_select_variable_with_top_k():
    inst = generate_sudoku(seed=12, difficulty="easy")
    model = LightGBMModel(seed=42)

    # With top_k = 3, chosen variable must be in top 3 MRV candidates
    top3 = get_top_k_mrv_candidates(inst.csp, k=3)
    chosen = model.select_variable(inst.csp, top_k=3)
    assert chosen in top3
