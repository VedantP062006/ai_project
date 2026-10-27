"""Unit tests for bitmask operations and domain utilities."""

import pytest
from solver.csp import bitmask_to_values, values_to_bitmask, CSP


def test_bitmask_conversions():
    values = [0, 2, 5, 8]
    mask = values_to_bitmask(values)
    assert mask == (1 << 0) | (1 << 2) | (1 << 5) | (1 << 8)
    recovered = bitmask_to_values(mask)
    assert recovered == values


def test_bitmask_empty_and_single():
    assert bitmask_to_values(0) == []
    assert values_to_bitmask([]) == 0

    mask_single = values_to_bitmask([4])
    assert mask_single == (1 << 4)
    assert bitmask_to_values(mask_single) == [4]


def test_csp_domain_operations():
    init_domains = [values_to_bitmask([0, 1, 2]), values_to_bitmask([1])]
    csp = CSP(num_variables=2, initial_domains=init_domains)

    assert not csp.is_assigned(0)
    assert csp.is_assigned(1)
    assert csp.assigned_value(1) == 1
    assert csp.assigned_value(0) is None
    assert csp.domain_size(0) == 3
    assert csp.domain_size(1) == 1
    assert not csp.is_complete()

    # Assign variable 0
    csp.domains[0] = (1 << 2)
    assert csp.is_assigned(0)
    assert csp.assigned_value(0) == 2
    assert csp.is_complete()

    # Reset
    csp.reset_domains()
    assert not csp.is_assigned(0)
    assert csp.domain_size(0) == 3
