"""Constraint propagation engines: AC-3 maintaining arc consistency (MAC) and Forward Checking."""

from __future__ import annotations
from typing import Deque, Iterable, List, Optional, Sequence, Set, Tuple, Any
from collections import deque


def ac3_propagate(
    csp: Any,
    arcs: Optional[Iterable[Tuple[int, int]]] = None,
    trail: Optional[List[Tuple[int, int]]] = None,
    track_weights: bool = True,
) -> bool:
    """
    AC-3 constraint propagation algorithm with bitmask domain updates.
    
    Args:
        csp: CSP instance to propagate.
        arcs: Optional initial iterable of directed (var_i, var_j) arcs.
              If None, initializes with all binary constraint arcs.
        trail: List to accumulate (var, pruned_mask) tuples for undoing.
        track_weights: If True, increments constraint weights upon wipeout (for dom/wdeg).
        
    Returns:
        True if arc consistency was established without domain wipeout, False otherwise.
    """
    if trail is None:
        trail = []

    queue: Deque[Tuple[int, int]] = deque()
    in_queue: Set[Tuple[int, int]] = set()

    if arcs is None:
        for (u, v) in csp.binary_constraints.keys():
            queue.append((u, v))
            in_queue.add((u, v))
    else:
        for arc in arcs:
            if arc not in in_queue:
                queue.append(arc)
                in_queue.add(arc)

    while queue:
        var_i, var_j = queue.popleft()
        in_queue.remove((var_i, var_j))

        # Constraints between var_i and var_j
        constraints = csp.binary_constraints.get((var_i, var_j), [])
        domain_revised = False

        for c in constraints:
            revised = c.revise_arc(csp, var_i, var_j, trail)
            if revised:
                domain_revised = True
                if csp.domains[var_i] == 0:
                    # Domain wipeout!
                    if track_weights:
                        c.record_failure()
                    return False

        if domain_revised:
            # Re-queue all neighbors of var_i except var_j
            for k in csp.neighbors[var_i]:
                if k != var_j:
                    arc = (k, var_i)
                    if arc not in in_queue:
                        queue.append(arc)
                        in_queue.add(arc)

    # Optional fast propagation for registered AllDifferent groups
    for group in csp.all_diff_groups:
        assigned_mask = 0
        for v in group:
            mask = csp.domains[v]
            if mask == 0:
                return False
            if mask.bit_count() == 1:
                if (assigned_mask & mask) != 0:
                    return False  # Direct conflict
                assigned_mask |= mask

        if assigned_mask > 0:
            for v in group:
                mask = csp.domains[v]
                if mask.bit_count() > 1 and (mask & assigned_mask) != 0:
                    pruned = mask & assigned_mask
                    new_mask = mask ^ pruned
                    if new_mask == 0:
                        return False
                    csp.domains[v] = new_mask
                    trail.append((v, pruned))

    return True


def forward_checking(
    csp: Any,
    assigned_var: int,
    trail: List[Tuple[int, int]],
    track_weights: bool = True,
) -> bool:
    """
    Forward Checking propagation: prunes only direct neighbors of the assigned variable.
    
    Args:
        csp: CSP instance.
        assigned_var: Variable that was just assigned.
        trail: Undo trail list.
        track_weights: Whether to update constraint weights on wipeout.
        
    Returns:
        True if all neighbor domains remain non-empty, False on wipeout.
    """
    for neighbor in csp.neighbors[assigned_var]:
        constraints = csp.binary_constraints.get((neighbor, assigned_var), [])
        for c in constraints:
            revised = c.revise_arc(csp, neighbor, assigned_var, trail)
            if revised and csp.domains[neighbor] == 0:
                if track_weights:
                    c.record_failure()
                return False

    return True
