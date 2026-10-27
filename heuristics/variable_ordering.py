"""Classical variable ordering heuristics and candidate selection utilities."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any
import random
from solver.csp import CSP


def var_static(csp: CSP, model: Optional[Any] = None, timer: Optional[Any] = None) -> int:
    """Static variable ordering: choose first unassigned variable by index."""
    for v in range(csp.num_variables):
        if not csp.is_assigned(v):
            return v
    return -1


def var_random(csp: CSP, model: Optional[Any] = None, timer: Optional[Any] = None) -> int:
    """Random variable ordering: pick uniformly among unassigned variables."""
    unassigned = [v for v in range(csp.num_variables) if not csp.is_assigned(v)]
    if not unassigned:
        return -1
    return random.choice(unassigned)


def var_mrv(csp: CSP, model: Optional[Any] = None, timer: Optional[Any] = None) -> int:
    """
    Minimum Remaining Values (MRV) heuristic:
    Select unassigned variable with smallest domain size.
    """
    best_var = -1
    min_size = 999999
    for v in range(csp.num_variables):
        if not csp.is_assigned(v):
            sz = csp.domain_size(v)
            if sz < min_size:
                min_size = sz
                best_var = v
                if sz == 2:  # Minimum possible for unassigned
                    break
    return best_var


def var_degree(csp: CSP, model: Optional[Any] = None, timer: Optional[Any] = None) -> int:
    """
    Degree heuristic:
    Select unassigned variable with maximum number of constraints with other unassigned variables.
    """
    best_var = -1
    max_deg = -1
    for v in range(csp.num_variables):
        if not csp.is_assigned(v):
            # Dynamic degree: count of unassigned neighbors
            deg = sum(1 for n in csp.neighbors[v] if not csp.is_assigned(n))
            if deg > max_deg:
                max_deg = deg
                best_var = v
    return best_var


def var_mrv_degree(csp: CSP, model: Optional[Any] = None, timer: Optional[Any] = None) -> int:
    """
    MRV with Degree tie-breaking:
    Minimize domain size; break ties by maximizing dynamic degree.
    """
    best_var = -1
    min_size = 999999
    max_deg = -1

    for v in range(csp.num_variables):
        if not csp.is_assigned(v):
            sz = csp.domain_size(v)
            deg = sum(1 for n in csp.neighbors[v] if not csp.is_assigned(n))
            if (sz < min_size) or (sz == min_size and deg > max_deg):
                min_size = sz
                max_deg = deg
                best_var = v

    return best_var


def var_dom_wdeg(csp: CSP, model: Optional[Any] = None, timer: Optional[Any] = None) -> int:
    """
    Domain over Weighted Degree (dom/wdeg) heuristic (Boussemart et al., 2004).
    
    Selects unassigned variable minimizing |D(v)| / wdeg(v),
    where wdeg(v) is the sum of weights of constraints involving v and at least one other unassigned variable.
    """
    best_var = -1
    best_ratio = float("inf")

    for v in range(csp.num_variables):
        if not csp.is_assigned(v):
            sz = csp.domain_size(v)
            # Compute wdeg(v)
            wdeg = 0.0
            for neighbor in csp.neighbors[v]:
                if not csp.is_assigned(neighbor):
                    constraints = csp.binary_constraints.get((v, neighbor), [])
                    for c in constraints:
                        wdeg += c.weight

            # Also check AllDifferent groups involving v with other unassigned variables
            for group in csp.all_diff_groups:
                if v in group:
                    unassigned_in_group = sum(1 for u in group if not csp.is_assigned(u))
                    if unassigned_in_group > 1:
                        wdeg += 1.0

            wdeg = max(wdeg, 1e-5)
            ratio = sz / wdeg
            if ratio < best_ratio:
                best_ratio = ratio
                best_var = v

    return best_var


def get_top_k_mrv_candidates(csp: CSP, k: int = 3) -> List[int]:
    """
    Extract the top-k candidate variables prioritized by MRV and dynamic degree.
    Used for H4 hybrid tie-breaking with ML models.
    """
    candidates = []
    for v in range(csp.num_variables):
        if not csp.is_assigned(v):
            sz = csp.domain_size(v)
            deg = sum(1 for n in csp.neighbors[v] if not csp.is_assigned(n))
            candidates.append((sz, -deg, v))

    candidates.sort()
    return [v for _, _, v in candidates[:k]]


VARIABLE_HEURISTICS: Dict[str, Callable] = {
    "static": var_static,
    "random": var_random,
    "mrv": var_mrv,
    "degree": var_degree,
    "mrv_degree": var_mrv_degree,
    "dom_wdeg": var_dom_wdeg,
}
