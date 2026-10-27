"""CSP representation using high-performance bitmask domains."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any, Sequence
from dataclasses import dataclass, field


def bitmask_to_values(mask: int) -> List[int]:
    """Convert an integer bitmask into a list of value indices."""
    vals: List[int] = []
    while mask > 0:
        lsb = mask & -mask
        vals.append(lsb.bit_length() - 1)
        mask ^= lsb
    return vals


def values_to_bitmask(values: Sequence[int]) -> int:
    """Convert a sequence of integer values to a bitmask."""
    mask = 0
    for v in values:
        if v < 0:
            raise ValueError(f"Domain values must be non-negative integers, got {v}")
        mask |= (1 << v)
    return mask


class CSP:
    """
    Constraint Satisfaction Problem representation with bitmask domains.
    
    Variables are 0-indexed integers [0, num_variables - 1].
    Domains are stored as integer bitmasks where bit `v` is 1 if value `v` is active.
    """

    def __init__(
        self,
        num_variables: int,
        initial_domains: Sequence[int],
        variable_names: Optional[Sequence[str]] = None,
        name: str = "CSP",
    ) -> None:
        self.num_variables = num_variables
        self.initial_domains = list(initial_domains)
        self.domains = list(initial_domains)
        self.name = name

        if variable_names is not None:
            if len(variable_names) != num_variables:
                raise ValueError("variable_names length must match num_variables")
            self.variable_names = list(variable_names)
        else:
            self.variable_names = [f"X{i}" for i in range(num_variables)]

        self.neighbors: List[Set[int]] = [set() for _ in range(num_variables)]
        self.constraints: List[Any] = []
        self.binary_constraints: Dict[Tuple[int, int], List[Any]] = {}
        self.all_diff_groups: List[List[int]] = []

    def add_constraint(self, constraint: Any) -> None:
        """Register a constraint and update graph adjacency."""
        self.constraints.append(constraint)
        scope = constraint.scope
        if len(scope) == 2:
            u, v = scope
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)
            self.binary_constraints.setdefault((u, v), []).append(constraint)
            self.binary_constraints.setdefault((v, u), []).append(constraint)
        elif len(scope) > 2:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    self.neighbors[scope[i]].add(scope[j])
                    self.neighbors[scope[j]].add(scope[i])

    def add_all_diff_group(self, group: Sequence[int]) -> None:
        """Register an all-different group for high-performance propagation."""
        var_list = list(group)
        self.all_diff_groups.append(var_list)
        for i in range(len(var_list)):
            for j in range(i + 1, len(var_list)):
                self.neighbors[var_list[i]].add(var_list[j])
                self.neighbors[var_list[j]].add(var_list[i])

    def is_assigned(self, var: int) -> bool:
        """Check if variable has a single assigned value."""
        return self.domains[var].bit_count() == 1

    def is_wiped_out(self, var: int) -> bool:
        """Check if variable domain is empty (wipeout)."""
        return self.domains[var] == 0

    def assigned_value(self, var: int) -> Optional[int]:
        """Return assigned value if assigned, else None."""
        mask = self.domains[var]
        if mask.bit_count() == 1:
            return (mask & -mask).bit_length() - 1
        return None

    def domain_size(self, var: int) -> int:
        """Number of remaining allowed values."""
        return self.domains[var].bit_count()

    def domain_values(self, var: int) -> List[int]:
        """List of active domain values."""
        return bitmask_to_values(self.domains[var])

    def is_complete(self) -> bool:
        """True if every variable is assigned and no wipeout exists."""
        return all(self.domains[i].bit_count() == 1 for i in range(self.num_variables))

    def reset_domains(self) -> None:
        """Reset mutable domains back to initial_domains."""
        self.domains = list(self.initial_domains)

    def current_assignment(self) -> Dict[int, int]:
        """Extract dict of all currently assigned variables."""
        assignment: Dict[int, int] = {}
        for var in range(self.num_variables):
            val = self.assigned_value(var)
            if val is not None:
                assignment[var] = val
        return assignment

    def clone(self) -> CSP:
        """Lightweight clone of CSP with identical constraints but fresh domain list."""
        cloned = CSP(
            num_variables=self.num_variables,
            initial_domains=self.initial_domains,
            variable_names=self.variable_names,
            name=self.name,
        )
        cloned.domains = list(self.domains)
        cloned.neighbors = [set(n) for n in self.neighbors]
        cloned.constraints = list(self.constraints)
        cloned.binary_constraints = {k: list(v) for k, v in self.binary_constraints.items()}
        cloned.all_diff_groups = [list(g) for g in self.all_diff_groups]
        return cloned
