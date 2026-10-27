"""Unit tests ensuring complete search determinism and reproducibility given a seed."""

import pytest
from generators.sudoku import generate_sudoku
from generators.graph_coloring import generate_graph_coloring
from heuristics.variable_ordering import var_random, var_mrv_degree
from heuristics.value_ordering import val_random, val_natural
from solver.backtracking import BacktrackingSolver


def test_search_determinism_mrv():
    inst1 = generate_sudoku(seed=55, difficulty="medium")
    inst2 = inst1.clone()

    solver = BacktrackingSolver(
        variable_heuristic=var_mrv_degree,
        value_heuristic=val_natural,
        propagation_mode="mac",
        time_limit_sec=5.0,
    )

    sol1, stats1 = solver.solve(inst1.csp, seed=42)
    sol2, stats2 = solver.solve(inst2.csp, seed=42)

    assert stats1.solved is True
    assert stats2.solved is True
    assert stats1.nodes == stats2.nodes
    assert stats1.backtracks == stats2.backtracks
    assert sol1 == sol2


def test_search_determinism_random_heuristic():
    inst1 = generate_graph_coloring(num_vertices=18, num_colors=3, avg_degree=3.8, planted=True, seed=88)
    inst2 = inst1.clone()

    solver = BacktrackingSolver(
        variable_heuristic=var_random,
        value_heuristic=val_random,
        propagation_mode="mac",
        time_limit_sec=5.0,
    )

    sol1, stats1 = solver.solve(inst1.csp, seed=999)
    sol2, stats2 = solver.solve(inst2.csp, seed=999)

    assert stats1.solved == stats2.solved
    assert stats1.nodes == stats2.nodes
    assert stats1.backtracks == stats2.backtracks
    assert sol1 == sol2
