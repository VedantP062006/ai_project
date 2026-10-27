"""Solution verification module: verifies solutions against all original constraints."""

from __future__ import annotations
from typing import Dict, Any


def verify_solution(csp: Any, assignment: Dict[int, int], raise_on_failure: bool = True) -> bool:
    """
    Independently verify an assignment against all constraints of a CSP.
    
    Args:
        csp: The CSP instance.
        assignment: Dict mapping variable index to integer value.
        raise_on_failure: If True, raises AssertionError with diagnostic message on failure.
        
    Returns:
        True if the assignment is complete and valid.
    """
    # 1. Completeness check
    for var in range(csp.num_variables):
        if var not in assignment:
            msg = f"Solution incomplete: variable {var} ({csp.variable_names[var]}) is unassigned."
            if raise_on_failure:
                raise AssertionError(msg)
            return False

        val = assignment[var]
        # 2. Domain check
        init_mask = csp.initial_domains[var]
        if (init_mask & (1 << val)) == 0:
            msg = (
                f"Domain violation: variable {var} ({csp.variable_names[var]}) assigned {val}, "
                f"which is not in initial domain bitmask {bin(init_mask)}."
            )
            if raise_on_failure:
                raise AssertionError(msg)
            return False

    # 3. Explicit constraints check
    for c in csp.constraints:
        if not c.check(assignment):
            msg = (
                f"Constraint violation: constraint {c.name} (scope={c.scope}) failed "
                f"on assignment values {[assignment.get(v) for v in c.scope]}."
            )
            if raise_on_failure:
                raise AssertionError(msg)
            return False

    # 4. AllDifferent groups check
    for group in csp.all_diff_groups:
        vals = [assignment[v] for v in group]
        if len(vals) != len(set(vals)):
            msg = f"AllDifferent group violation: variables {group} have non-unique values: {vals}."
            if raise_on_failure:
                raise AssertionError(msg)
            return False

    return True
