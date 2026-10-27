"""External ground-truth sanity check using Google OR-Tools (CP-SAT).

This module is strictly an external validation oracle for verifying instance satisfiability
and generator correctness during testing. It is NEVER part of the learned backtracking solver.
"""

from __future__ import annotations
from typing import Dict, Optional, Any
from ortools.sat.python import cp_model
from solver.csp import CSP, bitmask_to_values
from solver.constraints import BinaryNotEqualConstraint, BinaryTableConstraint, AllDifferentConstraint


def ortools_solve(csp: CSP, time_limit_sec: float = 10.0) -> Optional[Dict[int, int]]:
    """
    Solve a CSP instance using OR-Tools CP-SAT as an external ground-truth oracle.
    
    Returns:
        Dict mapping variable indices to values if satisfiable, else None.
    """
    model = cp_model.CpModel()
    cp_vars = {}

    for v in range(csp.num_variables):
        allowed_vals = bitmask_to_values(csp.initial_domains[v])
        if not allowed_vals:
            return None
        cp_vars[v] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(allowed_vals),
            name=f"X_{v}",
        )

    # Add constraints
    for c in csp.constraints:
        if isinstance(c, BinaryNotEqualConstraint):
            model.Add(cp_vars[c.var_a] != cp_vars[c.var_b])
        elif isinstance(c, BinaryTableConstraint):
            table = list(c.compatible_pairs)
            model.AddAllowedAssignments([cp_vars[c.var_a], cp_vars[c.var_b]], table)
        elif isinstance(c, AllDifferentConstraint):
            model.AddAllDifferent([cp_vars[v] for v in c.scope])

    for group in csp.all_diff_groups:
        model.AddAllDifferent([cp_vars[v] for v in group])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {v: solver.Value(cp_vars[v]) for v in range(csp.num_variables)}
    return None
