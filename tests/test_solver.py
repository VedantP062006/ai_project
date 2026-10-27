"""Unit tests for the BacktrackingSolver and solution verifier."""

import pytest
from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint
from solver.backtracking import BacktrackingSolver
from solver.verifier import verify_solution
from generators.sudoku import parse_sudoku_string, generate_sudoku
from generators.graph_coloring import generate_graph_coloring
from heuristics.variable_ordering import var_mrv_degree
from heuristics.value_ordering import val_natural


def test_solver_known_sudoku():
    # A standard valid Sudoku puzzle
    puzzle_str = (
        "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
    )
    inst = parse_sudoku_string(puzzle_str, name="test_easy_sudoku")
    solver = BacktrackingSolver(
        variable_heuristic=var_mrv_degree,
        value_heuristic=val_natural,
        propagation_mode="mac",
        time_limit_sec=5.0,
    )

    sol, stats = solver.solve(inst.csp, seed=42)
    assert stats.solved is True
    assert sol is not None
    assert len(sol) == 81
    # Solution must pass independent verification
    assert verify_solution(inst.csp, sol, raise_on_failure=True) is True


def test_solver_graph_coloring():
    inst = generate_graph_coloring(num_vertices=20, num_colors=3, avg_degree=4.0, planted=True, seed=123)
    solver = BacktrackingSolver(
        variable_heuristic=var_mrv_degree,
        value_heuristic=val_natural,
        propagation_mode="mac",
        time_limit_sec=5.0,
    )
    sol, stats = solver.solve(inst.csp, seed=123)
    assert stats.solved is True
    assert sol is not None
    assert verify_solution(inst.csp, sol, raise_on_failure=True) is True


def test_solver_infeasible():
    # 2 variables, domain {0}, constraint X0 != X1 -> infeasible
    csp = CSP(num_variables=2, initial_domains=[values_to_bitmask([0]), values_to_bitmask([0])])
    csp.add_constraint(BinaryNotEqualConstraint(0, 1))

    solver = BacktrackingSolver(propagation_mode="mac", time_limit_sec=2.0)
    sol, stats = solver.solve(csp)
    assert sol is None
    assert stats.status == "INFEASIBLE"


def test_verifier_catches_invalid_solution():
    inst = generate_sudoku(seed=99, difficulty="easy")
    solver = BacktrackingSolver(time_limit_sec=5.0)
    sol, stats = solver.solve(inst.csp)
    assert sol is not None

    # Corrupt solution by setting two adjacent cells in row 0 to the same value
    corrupted = dict(sol)
    corrupted[0] = corrupted[1]

    with pytest.raises(AssertionError):
        verify_solution(inst.csp, corrupted, raise_on_failure=True)
