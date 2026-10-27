"""Unit tests for problem generators and benchmark loaders."""

import pytest
from generators.sudoku import generate_sudoku, create_16x16_sudoku, get_published_hard_benchmarks
from generators.graph_coloring import generate_graph_coloring
from generators.cryptarithmetic import get_classic_cryptarithmetic
from generators.split import split_instances
from solver.sanity_check import ortools_solve
from solver.verifier import verify_solution


def test_sudoku_generator_and_ortools_sanity():
    inst = generate_sudoku(seed=42, difficulty="easy")
    assert inst.csp.num_variables == 81

    # Verify satisfiability via OR-Tools external oracle
    solution = ortools_solve(inst.csp, time_limit_sec=5.0)
    assert solution is not None
    assert verify_solution(inst.csp, solution, raise_on_failure=True) is True


def test_16x16_sudoku_structure():
    inst = create_16x16_sudoku()
    assert inst.csp.num_variables == 256
    assert len(inst.csp.all_diff_groups) == 16 + 16 + 16  # 16 rows, 16 cols, 16 blocks


def test_published_hard_sudoku_benchmarks():
    puzzles = get_published_hard_benchmarks()
    assert len(puzzles) >= 3
    for p in puzzles:
        assert p.csp.num_variables == 81


def test_graph_coloring_generator():
    inst = generate_graph_coloring(num_vertices=25, num_colors=3, avg_degree=4.0, planted=True, seed=77)
    assert inst.csp.num_variables == 25
    # Planted graph must be satisfiable
    solution = ortools_solve(inst.csp, time_limit_sec=5.0)
    assert solution is not None
    assert verify_solution(inst.csp, solution, raise_on_failure=True) is True


def test_cryptarithmetic_generator():
    inst = get_classic_cryptarithmetic("send_more_money")
    assert inst.csp.num_variables >= 8
    # OR-Tools sanity check
    solution = ortools_solve(inst.csp, time_limit_sec=5.0)
    assert solution is not None


def test_dataset_split_isolation():
    instances = [generate_sudoku(seed=i, difficulty="medium", name=f"sudoku_{i}") for i in range(20)]
    train, val, test = split_instances(instances, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3

    train_names = set(x.name for x in train)
    val_names = set(x.name for x in val)
    test_names = set(x.name for x in test)

    assert len(train_names & val_names) == 0
    assert len(train_names & test_names) == 0
    assert len(val_names & test_names) == 0
