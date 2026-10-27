"""Value ordering heuristics for CSP backtracking search."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Any
import random
from solver.csp import CSP


def val_natural(
    csp: CSP,
    var: int,
    model: Optional[Any] = None,
    timer: Optional[Any] = None,
) -> List[int]:
    """Natural / lexicographic value ordering: smallest index first."""
    return csp.domain_values(var)


def val_random(
    csp: CSP,
    var: int,
    model: Optional[Any] = None,
    timer: Optional[Any] = None,
) -> List[int]:
    """Random value ordering: shuffle domain values."""
    vals = csp.domain_values(var)
    random.shuffle(vals)
    return vals


def val_lcv(
    csp: CSP,
    var: int,
    model: Optional[Any] = None,
    timer: Optional[Any] = None,
) -> List[int]:
    """
    Least Constraining Value (LCV) heuristic:
    Order values ascending by the number of values they rule out in unassigned neighbors' domains.
    """
    values = csp.domain_values(var)
    if len(values) <= 1:
        return values

    val_penalties: List[Tuple[int, int]] = []

    for val in values:
        penalty = 0
        val_mask = 1 << val

        for neighbor in csp.neighbors[var]:
            if not csp.is_assigned(neighbor):
                n_mask = csp.domains[neighbor]
                constraints = csp.binary_constraints.get((neighbor, var), [])
                if not constraints:
                    # Generic inequality if in same all-diff group
                    if (n_mask & val_mask) != 0:
                        penalty += 1
                else:
                    for c in constraints:
                        # Check how many values in neighbor lack support with var = val
                        if hasattr(c, "support_a_for_b"):
                            # BinaryTableConstraint
                            if c.var_a == neighbor and c.var_b == var:
                                allowed_in_n = c.support_a_for_b[val]
                                penalty += (n_mask & ~allowed_in_n).bit_count()
                            elif c.var_b == neighbor and c.var_a == var:
                                allowed_in_n = c.support_b_for_a[val]
                                penalty += (n_mask & ~allowed_in_n).bit_count()
                        else:
                            # Inequality
                            if (n_mask & val_mask) != 0:
                                penalty += 1

        val_penalties.append((penalty, val))

    # Sort ascending: least constraining value first
    val_penalties.sort(key=lambda item: item[0])
    return [v for _, v in val_penalties]


VALUE_HEURISTICS: Dict[str, Callable] = {
    "natural": val_natural,
    "random": val_random,
    "lcv": val_lcv,
}
