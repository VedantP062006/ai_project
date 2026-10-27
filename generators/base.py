"""Base definitions for CSP problem generators and benchmark instances."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from solver.csp import CSP


@dataclass
class CSPInstance:
    """Standardized benchmark instance wrapper."""

    name: str
    family: str  # "sudoku", "graph_coloring", "cryptarithmetic", "timetabling"
    difficulty: str  # "easy", "medium", "hard"
    csp: CSP
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> CSPInstance:
        """Deep copy instance with independent CSP domains."""
        return CSPInstance(
            name=self.name,
            family=self.family,
            difficulty=self.difficulty,
            csp=self.csp.clone(),
            metadata=dict(self.metadata),
        )
