"""Core CSP representation and solver package."""

from solver.csp import CSP
from solver.constraints import Constraint, BinaryConstraint, BinaryNotEqualConstraint, BinaryTableConstraint, AllDifferentConstraint
from solver.propagation import ac3_propagate, forward_checking
from solver.instrumentation import SearchStats
from solver.verifier import verify_solution
from solver.backtracking import BacktrackingSolver

__all__ = [
    "CSP",
    "Constraint",
    "BinaryConstraint",
    "BinaryNotEqualConstraint",
    "BinaryTableConstraint",
    "AllDifferentConstraint",
    "ac3_propagate",
    "forward_checking",
    "SearchStats",
    "verify_solution",
    "BacktrackingSolver",
]
