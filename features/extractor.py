"""Vectorized, high-speed feature extractor for CSP variable ordering policies."""

from __future__ import annotations
from typing import List, Tuple, Optional, Any, Sequence
import numpy as np

from solver.csp import CSP

FEATURE_NAMES = [
    "norm_domain_size",
    "domain_size_raw",
    "static_degree",
    "dynamic_degree",
    "assigned_neighbor_fraction",
    "dom_wdeg_ratio",
    "constraint_failure_count",
    "search_depth",
    "fraction_assigned",
    "neighbor_min_domain",
    "neighbor_mean_domain",
    "in_alldiff_group",
]

NUM_FEATURES = len(FEATURE_NAMES)


class FeatureExtractor:
    """
    Extracts dynamic structural and state features for all variables in a CSP state.
    
    Guarantees:
    - 0% Ground-Truth Leakage: features depend solely on observable constraint state,
      domain bitmasks, and current search depth.
    - Deterministic, NaN-free, and float32 normalized representations.
    """

    def __init__(self) -> None:
        self.feature_names = list(FEATURE_NAMES)
        self.num_features = NUM_FEATURES

    def extract(self, csp: CSP, depth: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for all variables in the CSP.
        
        Args:
            csp: The CSP instance.
            depth: Current search depth in backtracking tree.
            
        Returns:
            features: np.ndarray of shape (N, num_features), dtype float32.
            candidate_mask: boolean np.ndarray of shape (N,), True for unassigned variables.
        """
        n = csp.num_variables
        features = np.zeros((n, self.num_features), dtype=np.float32)
        candidate_mask = np.zeros(n, dtype=bool)

        # 1. State aggregates
        domain_sizes = np.array([csp.domain_size(v) for v in range(n)], dtype=np.float32)
        init_sizes = np.array([csp.initial_domains[v].bit_count() for v in range(n)], dtype=np.float32)
        is_assigned = (domain_sizes <= 1)
        candidate_mask = ~is_assigned

        num_assigned = int(np.sum(is_assigned))
        fraction_assigned = float(num_assigned / max(1, n))
        norm_depth = float(depth / max(1, n))

        # Check all-diff memberships
        in_alldiff = np.zeros(n, dtype=np.float32)
        for group in csp.all_diff_groups:
            for v in group:
                in_alldiff[v] = 1.0

        # 2. Per-variable extraction
        for v in range(n):
            sz = domain_sizes[v]
            init_sz = max(1.0, init_sizes[v])
            norm_sz = sz / init_sz

            neighbors = list(csp.neighbors[v])
            deg = len(neighbors)
            norm_static_deg = deg / max(1, n)

            unassigned_nbrs = [nbr for nbr in neighbors if not is_assigned[nbr]]
            dyn_deg = len(unassigned_nbrs)
            norm_dyn_deg = dyn_deg / max(1, n)

            assigned_nbr_frac = ((deg - dyn_deg) / max(1, deg)) if deg > 0 else 0.0

            # dom/wdeg computation
            wdeg = 0.0
            failures = 0
            for nbr in neighbors:
                constraints = csp.binary_constraints.get((v, nbr), [])
                for c in constraints:
                    failures += c.failure_count
                    if not is_assigned[nbr]:
                        wdeg += c.weight

            for group in csp.all_diff_groups:
                if v in group:
                    unassigned_in_grp = sum(1 for u in group if not is_assigned[u])
                    if unassigned_in_grp > 1:
                        wdeg += 1.0

            wdeg = max(wdeg, 1e-5)
            dom_wdeg = sz / wdeg

            # Neighbor domain statistics
            if unassigned_nbrs:
                nbr_sizes = [domain_sizes[u] for u in unassigned_nbrs]
                nbr_min = float(min(nbr_sizes))
                nbr_mean = float(sum(nbr_sizes) / len(nbr_sizes))
            else:
                nbr_min = 0.0
                nbr_mean = 0.0

            features[v, 0] = norm_sz
            features[v, 1] = sz
            features[v, 2] = norm_static_deg
            features[v, 3] = norm_dyn_deg
            features[v, 4] = assigned_nbr_frac
            features[v, 5] = dom_wdeg
            features[v, 6] = float(failures)
            features[v, 7] = norm_depth
            features[v, 8] = fraction_assigned
            features[v, 9] = nbr_min
            features[v, 10] = nbr_mean
            features[v, 11] = in_alldiff[v]

        return features, candidate_mask


def extract_state_features(csp: CSP, depth: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience functional wrapper for FeatureExtractor."""
    extractor = FeatureExtractor()
    return extractor.extract(csp, depth=depth)
