"""Unit tests for classical variable and value ordering heuristics."""

import pytest
from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint
from heuristics.variable_ordering import (
    var_static,
    var_random,
    var_mrv,
    var_degree,
    var_mrv_degree,
    var_dom_wdeg,
    get_top_k_mrv_candidates,
)
from heuristics.value_ordering import val_natural, val_random, val_lcv


def test_variable_heuristics_selection():
    # 4 variables:
    # X0: domain size 3, degree 1
    # X1: domain size 2, degree 3
    # X2: domain size 4, degree 2
    # X3: domain size 1 (already assigned)
    init_domains = [
        values_to_bitmask([0, 1, 2]),
        values_to_bitmask([0, 1]),
        values_to_bitmask([0, 1, 2, 3]),
        values_to_bitmask([0]),
    ]
    csp = CSP(num_variables=4, initial_domains=init_domains)
    csp.add_constraint(BinaryNotEqualConstraint(1, 0))
    csp.add_constraint(BinaryNotEqualConstraint(1, 2))
    csp.add_constraint(BinaryNotEqualConstraint(1, 3))

    # Static picks first unassigned: X0
    assert var_static(csp) == 0

    # MRV picks smallest unassigned domain (size 2): X1
    assert var_mrv(csp) == 1

    # Degree picks maximum unassigned degree: X1 has 2 unassigned neighbors (0 and 2)
    assert var_degree(csp) == 1

    # MRV+Degree picks X1
    assert var_mrv_degree(csp) == 1

    # dom/wdeg picks X1
    assert var_dom_wdeg(csp) == 1

    # Top-k candidates
    top2 = get_top_k_mrv_candidates(csp, k=2)
    assert top2[0] == 1  # X1 has size 2
    assert top2[1] == 0  # X0 has size 3


def test_value_heuristics():
    csp = CSP(num_variables=2, initial_domains=[values_to_bitmask([2, 5, 8]), values_to_bitmask([2, 5])])
    csp.add_constraint(BinaryNotEqualConstraint(0, 1))

    # Natural ordering
    assert val_natural(csp, 0) == [2, 5, 8]

    # Random ordering is permutation
    shuffled = val_random(csp, 0)
    assert sorted(shuffled) == [2, 5, 8]

    # LCV ordering: value 8 does not conflict with X1's domain {2, 5},
    # so assigning 8 rules out 0 values in X1!
    # Assigning 2 or 5 rules out 1 value in X1.
    # Therefore LCV should prioritize 8!
    lcv_order = val_lcv(csp, 0)
    assert lcv_order[0] == 8
