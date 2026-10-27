"""Unit tests for the Bipartite Graph Neural Network policy."""

import pytest
import numpy as np
import torch
from pathlib import Path

from generators.sudoku import generate_sudoku
from generators.graph_coloring import generate_graph_coloring
from generators.cryptarithmetic import get_classic_cryptarithmetic
from models.gnn_model import GNNModel, build_bipartite_graph, GraphSearchStateSample
from training.gnn_training import collect_graph_trajectories, train_gnn_models
from solver.backtracking import BacktrackingSolver
from heuristics.value_ordering import VALUE_HEURISTICS


def test_bipartite_graph_construction_shapes():
    sudoku = generate_sudoku(seed=42, difficulty="easy")
    var_t, con_t, edge_var, edge_con, mask = build_bipartite_graph(sudoku.csp)

    assert var_t.shape == (81, 12)
    assert con_t.shape[1] == 7
    assert len(edge_var) == len(edge_con)
    assert len(edge_var) > 0
    assert edge_var.max().item() < 81
    assert edge_con.max().item() < con_t.shape[0]
    assert mask.shape == (81,)


def test_bipartite_graph_other_families():
    # Graph coloring
    gc = generate_graph_coloring(num_vertices=20, num_colors=3, seed=42)
    var_t, con_t, e_v, e_c, mask = build_bipartite_graph(gc.csp)
    assert var_t.shape == (20, 12)
    assert con_t.shape[1] == 7
    assert len(e_v) > 0

    # Cryptarithmetic
    crypto = get_classic_cryptarithmetic("send_more_money")
    var_t, con_t, e_v, e_c, mask = build_bipartite_graph(crypto.csp)
    assert var_t.shape == (crypto.csp.num_variables, 12)
    assert con_t.shape[1] == 7
    assert len(e_v) > 0


def test_gnn_forward_pass_and_ranking():
    gc = generate_graph_coloring(num_vertices=15, num_colors=3, seed=123)
    model = GNNModel(seed=42)

    scores = model.score_graph(gc.csp)
    assert scores.shape == (15,)
    assert not np.isnan(scores).any()
    assert model.last_inference_time > 0.0


def test_gnn_save_load_determinism(tmp_path):
    model1 = GNNModel(seed=42)
    gc = generate_graph_coloring(num_vertices=15, num_colors=3, seed=123)
    scores1 = model1.score_graph(gc.csp)

    save_file = tmp_path / "gnn_test.pt"
    model1.save(str(save_file))

    model2 = GNNModel(seed=999)
    model2.load(str(save_file))
    scores2 = model2.score_graph(gc.csp)

    np.testing.assert_allclose(scores1, scores2, atol=1e-6)


def test_gnn_training_and_solver_integration(tmp_path):
    train_inst = [generate_sudoku(seed=s, difficulty="easy") for s in range(2)]
    samples = collect_graph_trajectories(
        instances=train_inst,
        oracle_type="mrv",
        max_states_per_instance=5,
        top_k_candidates=3,
        seed=42,
    )
    assert len(samples) > 0

    trained = train_gnn_models(samples, seeds=[42], epochs=5, save_dir=str(tmp_path))
    assert len(trained) == 1
    model, _ = trained[0]

    # Test solver integration
    test_inst = generate_sudoku(seed=99, difficulty="medium")
    h_fn = model.create_heuristic_fn(top_k=3)

    solver = BacktrackingSolver(
        variable_heuristic=h_fn,
        value_heuristic=VALUE_HEURISTICS["natural"],
        verify=True,
    )
    sol, stats = solver.solve(test_inst.csp)
    assert stats.status == "SOLVED"
    assert sol is not None
    assert stats.nodes >= 0
