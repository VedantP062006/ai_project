"""Constraint definitions with fast bitmask arc consistency revision."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any
from abc import ABC, abstractmethod


class Constraint(ABC):
    """Abstract base class for all constraints."""

    def __init__(self, scope: Sequence[int], name: Optional[str] = None) -> None:
        self.scope: Tuple[int, ...] = tuple(scope)
        self.name: str = name or f"{self.__class__.__name__}_{'_'.join(map(str, self.scope))}"
        self.weight: float = 1.0  # dom/wdeg constraint weight
        self.failure_count: int = 0  # Count of domain wipeouts caused by this constraint

    @abstractmethod
    def check(self, assignment: Dict[int, int]) -> bool:
        """Verify whether an assignment satisfies this constraint."""
        pass

    @abstractmethod
    def revise_arc(
        self,
        csp: Any,
        var_i: int,
        var_j: int,
        trail: List[Tuple[int, int]],
    ) -> bool:
        """
        Revise the domain of var_i given var_j.
        
        Pushes (var_i, pruned_mask) to trail if any values are pruned.
        Returns True if domain was modified, False otherwise.
        """
        pass

    def record_failure(self) -> None:
        """Increment failure weight upon domain wipeout during search."""
        self.weight += 1.0
        self.failure_count += 1


class BinaryConstraint(Constraint):
    """Base class for binary constraints between two variables."""

    def __init__(self, var_a: int, var_b: int, name: Optional[str] = None) -> None:
        super().__init__(scope=(var_a, var_b), name=name)
        self.var_a = var_a
        self.var_b = var_b


class BinaryNotEqualConstraint(BinaryConstraint):
    """Binary inequality constraint: X_a != X_b."""

    def __init__(self, var_a: int, var_b: int, name: Optional[str] = None) -> None:
        super().__init__(var_a, var_b, name=name or f"X{var_a}!=X{var_b}")

    def check(self, assignment: Dict[int, int]) -> bool:
        val_a = assignment.get(self.var_a)
        val_b = assignment.get(self.var_b)
        if val_a is not None and val_b is not None:
            return val_a != val_b
        return True

    def revise_arc(
        self,
        csp: Any,
        var_i: int,
        var_j: int,
        trail: List[Tuple[int, int]],
    ) -> bool:
        # If var_j has a single value v, v cannot be in var_i's domain.
        mask_j = csp.domains[var_j]
        if mask_j.bit_count() == 1:
            mask_i = csp.domains[var_i]
            if (mask_i & mask_j) != 0:
                pruned = mask_i & mask_j
                csp.domains[var_i] ^= pruned
                trail.append((var_i, pruned))
                return True
        return False


class BinaryTableConstraint(BinaryConstraint):
    """
    Binary constraint defined by an explicit set of compatible (val_a, val_b) pairs.
    Precomputes bitmask support tables for O(1) support checks per value.
    """

    def __init__(
        self,
        var_a: int,
        var_b: int,
        compatible_pairs: Sequence[Tuple[int, int]],
        max_domain_size: int = 64,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(var_a, var_b, name=name)
        self.compatible_pairs = set(compatible_pairs)
        self.max_domain_size = max_domain_size

        # Precompute support bitmasks:
        # support_b_for_a[va]: bitmask of allowed vb values in B
        self.support_b_for_a: List[int] = [0] * max_domain_size
        # support_a_for_b[vb]: bitmask of allowed va values in A
        self.support_a_for_b: List[int] = [0] * max_domain_size

        for va, vb in self.compatible_pairs:
            if va < max_domain_size and vb < max_domain_size:
                self.support_b_for_a[va] |= (1 << vb)
                self.support_a_for_b[vb] |= (1 << va)

    def check(self, assignment: Dict[int, int]) -> bool:
        val_a = assignment.get(self.var_a)
        val_b = assignment.get(self.var_b)
        if val_a is not None and val_b is not None:
            return (val_a, val_b) in self.compatible_pairs
        return True

    def revise_arc(
        self,
        csp: Any,
        var_i: int,
        var_j: int,
        trail: List[Tuple[int, int]],
    ) -> bool:
        mask_i = csp.domains[var_i]
        mask_j = csp.domains[var_j]

        if mask_i == 0 or mask_j == 0:
            return False

        if var_i == self.var_a and var_j == self.var_b:
            support_table = self.support_b_for_a
        elif var_i == self.var_b and var_j == self.var_a:
            support_table = self.support_a_for_b
        else:
            return False

        pruned_mask = 0
        temp = mask_i
        while temp > 0:
            lsb = temp & -temp
            vi = lsb.bit_length() - 1
            temp ^= lsb
            # Has support if (support_table[vi] & mask_j) != 0
            if (support_table[vi] & mask_j) == 0:
                pruned_mask |= lsb

        if pruned_mask != 0:
            csp.domains[var_i] ^= pruned_mask
            trail.append((var_i, pruned_mask))
            return True

        return False


class AllDifferentConstraint(Constraint):
    """
    AllDifferent constraint over a subset of variables.
    Provides clique-level checking and propagation.
    """

    def __init__(self, variables: Sequence[int], name: Optional[str] = None) -> None:
        super().__init__(scope=variables, name=name or f"AllDiff_{len(variables)}")

    def check(self, assignment: Dict[int, int]) -> bool:
        assigned_vals: List[int] = []
        for var in self.scope:
            val = assignment.get(var)
            if val is not None:
                assigned_vals.append(val)
        return len(assigned_vals) == len(set(assigned_vals))

    def revise_arc(
        self,
        csp: Any,
        var_i: int,
        var_j: int,
        trail: List[Tuple[int, int]],
    ) -> bool:
        # Propagate: If var_j has a single value v, remove v from var_i
        mask_j = csp.domains[var_j]
        if mask_j.bit_count() == 1:
            mask_i = csp.domains[var_i]
            if (mask_i & mask_j) != 0:
                pruned = mask_i & mask_j
                csp.domains[var_i] ^= pruned
                trail.append((var_i, pruned))
                return True
        return False

    def propagate_group(self, csp: Any, trail: List[Tuple[int, int]]) -> bool:
        """
        Specialised high-performance group propagator.
        1. Fixed-value pruning: values assigned in 1-sized domains removed from others.
        2. Pigeonhole check: if union domain size < number of unassigned variables, wipeout!
        """
        assigned_mask = 0
        for var in self.scope:
            mask = csp.domains[var]
            if mask == 0:
                return False  # Already wiped out
            if mask.bit_count() == 1:
                if (assigned_mask & mask) != 0:
                    # Two variables have the same assigned value -> conflict!
                    return False
                assigned_mask |= mask

        # Prune assigned values from non-assigned variables in the group
        if assigned_mask > 0:
            for var in self.scope:
                mask = csp.domains[var]
                if mask.bit_count() > 1 and (mask & assigned_mask) != 0:
                    pruned = mask & assigned_mask
                    new_mask = mask ^ pruned
                    if new_mask == 0:
                        return False
                    csp.domains[var] = new_mask
                    trail.append((var, pruned))

        return True
