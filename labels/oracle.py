"""Expert oracle algorithms for variable and value ordering supervision."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np

from solver.csp import CSP
from solver.propagation import ac3_propagate


def mrv_imitation_oracle(csp: CSP, candidates: Sequence[int]) -> np.ndarray:
    """
    MRV imitation sanity oracle:
    Awards highest score (1.0) to candidate with smallest domain size,
    breaking ties by highest dynamic degree.
    """
    if not candidates:
        return np.array([], dtype=np.float32)

    scores = np.zeros(len(candidates), dtype=np.float32)
    ranked = []

    for idx, var in enumerate(candidates):
        sz = csp.domain_size(var)
        deg = sum(1 for n in csp.neighbors[var] if not csp.is_assigned(n))
        ranked.append((sz, -deg, idx))

    ranked.sort()
    # Best candidate gets 1.0, second gets 0.8, etc. (normalized exponential)
    for rank, (_, _, idx) in enumerate(ranked):
        scores[idx] = float(np.exp(-rank))

    return scores


def bounded_lookahead_oracle(
    csp: CSP,
    candidates: Sequence[int],
    lookahead_depth: int = 1,
) -> np.ndarray:
    """
    Strong-branching lookahead oracle:
    Simulates branching on each candidate variable with AC-3 propagation.
    Measures search-space pruning: average domain reductions and early wipeout detection.
    """
    if not candidates:
        return np.array([], dtype=np.float32)

    scores = np.zeros(len(candidates), dtype=np.float32)
    initial_unassigned_sizes = sum(csp.domain_size(v) for v in range(csp.num_variables) if not csp.is_assigned(v))

    for idx, var in enumerate(candidates):
        values = csp.domain_values(var)
        if not values:
            continue

        branch_reductions = []
        for val in values:
            trail: List[Tuple[int, int]] = []
            current_mask = csp.domains[var]
            pruned = current_mask ^ (1 << val)
            csp.domains[var] = (1 << val)
            trail.append((var, pruned))

            # Propagate
            arcs = [(k, var) for k in csp.neighbors[var]]
            ok = ac3_propagate(csp, arcs=arcs, trail=trail, track_weights=False)

            if not ok:
                # Immediate dead-end detected! Strong branching values this highly
                # as it eliminates an entire subtree.
                branch_reduction = initial_unassigned_sizes * 1.5
            else:
                remaining_sizes = sum(csp.domain_size(v) for v in range(csp.num_variables) if not csp.is_assigned(v))
                branch_reduction = max(0, initial_unassigned_sizes - remaining_sizes)

            branch_reductions.append(branch_reduction)

            # Undo trail
            while trail:
                u_var, u_mask = trail.pop()
                csp.domains[u_var] |= u_mask

        # Average reduction across all branches of this variable
        scores[idx] = float(np.mean(branch_reductions)) if branch_reductions else 0.0

    # Normalize scores to [0, 1]
    max_score = np.max(scores)
    min_score = np.min(scores)
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = np.ones_like(scores)

    return scores


def planted_value_oracle(
    csp: CSP,
    var: int,
    planted_solution: Dict[int, int],
) -> Optional[int]:
    """Return ground truth value from planted solution for value ordering supervision."""
    return planted_solution.get(var)
