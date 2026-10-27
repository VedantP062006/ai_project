"""Random Graph Colouring generator near the satisfiability phase transition."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import random
import networkx as nx

from solver.csp import CSP
from solver.constraints import BinaryNotEqualConstraint
from generators.base import CSPInstance


def generate_graph_coloring(
    num_vertices: int = 35,
    num_colors: int = 3,
    avg_degree: float = 4.5,  # Critical threshold for 3-coloring is ~4.69
    planted: bool = True,
    seed: int = 42,
    name: Optional[str] = None,
) -> CSPInstance:
    """
    Generate a random graph colouring CSP near the colourability threshold.
    
    If `planted=True`, partitions vertices into `num_colors` sets and only samples
    edges across partitions, guaranteeing satisfiability while maintaining phase-transition hardness.
    """
    rng = random.Random(seed)
    num_vars = num_vertices
    all_colors_mask = (1 << num_colors) - 1
    initial_domains = [all_colors_mask] * num_vars

    inst_name = name or f"col_v{num_vertices}_k{num_colors}_c{avg_degree:.1f}_s{seed}"

    csp = CSP(
        num_variables=num_vars,
        initial_domains=initial_domains,
        variable_names=[f"V{i}" for i in range(num_vars)],
        name=inst_name,
    )

    target_edges = int(round(num_vertices * avg_degree / 2.0))
    edges: List[Tuple[int, int]] = []
    planted_solution: Dict[int, int] = {}

    if planted:
        # Assign random color to each vertex
        for v in range(num_vars):
            planted_solution[v] = rng.randint(0, num_colors - 1)

        # Possible candidate cross-partition edges
        candidate_edges = []
        for u in range(num_vars):
            for v in range(u + 1, num_vars):
                if planted_solution[u] != planted_solution[v]:
                    candidate_edges.append((u, v))

        rng.shuffle(candidate_edges)
        edges = candidate_edges[: min(target_edges, len(candidate_edges))]
    else:
        # Standard G(n, m) near threshold
        all_possible = [(u, v) for u in range(num_vars) for v in range(u + 1, num_vars)]
        rng.shuffle(all_possible)
        edges = all_possible[: min(target_edges, len(all_possible))]

    # Add binary inequality constraints
    for u, v in edges:
        csp.add_constraint(BinaryNotEqualConstraint(u, v))

    difficulty = "easy" if num_vertices <= 25 else ("medium" if num_vertices <= 45 else "hard")

    return CSPInstance(
        name=inst_name,
        family="graph_coloring",
        difficulty=difficulty,
        csp=csp,
        metadata={
            "num_vertices": num_vertices,
            "num_colors": num_colors,
            "num_edges": len(edges),
            "avg_degree": (2 * len(edges)) / num_vertices,
            "planted": planted,
            "planted_solution": planted_solution if planted else None,
        },
    )
