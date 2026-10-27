"""Unit tests for the DAgger active learning loop."""

import pytest
from generators.sudoku import generate_sudoku
from models.lightgbm_model import LightGBMModel
from labels.dataset import collect_training_trajectories
from training.dagger import run_dagger_iteration, train_dagger


def test_run_dagger_iteration_collects_states():
    inst = [generate_sudoku(seed=10, difficulty="easy")]
    model = LightGBMModel(seed=42)

    samples = run_dagger_iteration(
        model=model,
        instances=inst,
        top_k=3,
        lookahead_depth=1,
        max_states_per_instance=5,
        beta=0.5,
        seed=42,
    )

    assert len(samples) > 0
    assert len(samples[0].candidate_indices) <= 3


def test_train_dagger_loop():
    train_inst = [generate_sudoku(seed=i, difficulty="easy") for i in range(2)]
    val_inst = [generate_sudoku(seed=99, difficulty="easy")]

    initial_samples = collect_training_trajectories(
        instances=train_inst,
        oracle_type="mrv",
        max_states_per_instance=5,
        top_k_candidates=3,
        seed=42,
    )

    model, history = train_dagger(
        train_instances=train_inst,
        val_instances=val_inst,
        initial_samples=initial_samples,
        model_type="lightgbm",
        num_iterations=2,
        top_k=3,
        seed=42,
        save_dir=None,
    )

    assert len(history) == 3  # Iteration 0, 1, 2
    assert history[-1]["dataset_size"] >= history[0]["dataset_size"]
