"""Unit tests for AC-3 propagation and Forward Checking."""

import pytest
from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint, BinaryTableConstraint, AllDifferentConstraint
from solver.propagation import ac3_propagate, forward_checking


def test_ac3_binary_inequality_pruning():
    # X0 in {1}, X1 in {1, 2} with X0 != X1.
    # Propagation should prune 1 from X1, leaving X1 in {2}.
    csp = CSP(num_variables=2, initial_domains=[values_to_bitmask([1]), values_to_bitmask([1, 2])])
    csp.add_constraint(BinaryNotEqualConstraint(0, 1))

    trail = []
    ok = ac3_propagate(csp, trail=trail)
    assert ok is True
    assert csp.domain_values(1) == [2]
    assert len(trail) == 1
    assert trail[0][0] == 1  # Variable 1 was pruned


def test_ac3_wipeout_detection():
    # X0 in {1}, X1 in {1} with X0 != X1. Impossible -> wipeout!
    csp = CSP(num_variables=2, initial_domains=[values_to_bitmask([1]), values_to_bitmask([1])])
    c = BinaryNotEqualConstraint(0, 1)
    csp.add_constraint(c)

    trail = []
    ok = ac3_propagate(csp, trail=trail, track_weights=True)
    assert ok is False
    assert c.failure_count == 1
    assert c.weight == 2.0


def test_ac3_binary_table_constraint():
    # X0 in {0, 1}, X1 in {0, 1}
    # Compatible: (0, 1) only.
    csp = CSP(num_variables=2, initial_domains=[values_to_bitmask([0, 1]), values_to_bitmask([0, 1])])
    c = BinaryTableConstraint(0, 1, [(0, 1)])
    csp.add_constraint(c)

    trail = []
    ok = ac3_propagate(csp, trail=trail)
    assert ok is True
    assert csp.domain_values(0) == [0]
    assert csp.domain_values(1) == [1]


def test_forward_checking():
    csp = CSP(num_variables=3, initial_domains=[values_to_bitmask([0, 1]), values_to_bitmask([0, 1]), values_to_bitmask([0, 1])])
    csp.add_constraint(BinaryNotEqualConstraint(0, 1))
    csp.add_constraint(BinaryNotEqualConstraint(1, 2))

    trail = []
    # Assign X0 <- 0
    csp.domains[0] = (1 << 0)
    ok = forward_checking(csp, assigned_var=0, trail=trail)
    assert ok is True
    # X1 should not have 0
    assert csp.domain_values(1) == [1]
    # X2 should NOT be pruned by FC (only direct neighbors pruned)
    assert csp.domain_values(2) == [0, 1]
