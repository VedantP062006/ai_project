"""Trajectory collector and dataset storage for imitation learning."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
import pickle
import numpy as np

from solver.csp import CSP
from solver.propagation import ac3_propagate
from features.extractor import FeatureExtractor
from labels.oracle import mrv_imitation_oracle, bounded_lookahead_oracle
from heuristics.variable_ordering import get_top_k_mrv_candidates
from generators.base import CSPInstance


@dataclass
class SearchStateSample:
    """A single labeled decision state during CSP search."""

    instance_name: str
    depth: int
    candidate_indices: List[int]
    candidate_features: np.ndarray  # Shape: (K, num_features)
    target_scores: np.ndarray       # Shape: (K,)
    best_candidate_idx: int


def collect_training_trajectories(
    instances: Sequence[CSPInstance],
    oracle_type: str = "lookahead",
    max_states_per_instance: int = 40,
    lookahead_depth: int = 1,
    top_k_candidates: Optional[int] = 5,
    seed: int = 42,
) -> List[SearchStateSample]:
    """
    Collect labeled search states across training instances.
    
    Args:
        instances: List of training CSP instances.
        oracle_type: 'lookahead' (strong branching) or 'mrv' (sanity).
        max_states_per_instance: Limit states sampled per instance to balance dataset.
        lookahead_depth: Lookahead depth for strong branching.
        top_k_candidates: If set, evaluates oracle on top-k MRV candidates (efficient for H4).
        seed: Random seed.
        
    Returns:
        List of SearchStateSample objects.
    """
    np.random.seed(seed)
    feature_extractor = FeatureExtractor()
    dataset: List[SearchStateSample] = []

    for inst in instances:
        csp = inst.csp.clone()
        trail = []
        # Initial propagation
        ok = ac3_propagate(csp, trail=trail)
        if not ok or csp.is_complete():
            continue

        states_collected = 0

        # Traverse search tree while collecting states
        def search_collect(depth: int) -> bool:
            nonlocal states_collected
            if csp.is_complete() or states_collected >= max_states_per_instance:
                return True

            # Extract features of the entire CSP
            features, candidate_mask = feature_extractor.extract(csp, depth=depth)
            unassigned_vars = np.where(candidate_mask)[0]
            if len(unassigned_vars) == 0:
                return True

            if top_k_candidates is not None and len(unassigned_vars) > top_k_candidates:
                candidates = get_top_k_mrv_candidates(csp, k=top_k_candidates)
            else:
                candidates = list(unassigned_vars)

            # Query oracle
            if oracle_type == "mrv":
                scores = mrv_imitation_oracle(csp, candidates)
            else:
                scores = bounded_lookahead_oracle(csp, candidates, lookahead_depth=lookahead_depth)

            best_local_idx = int(np.argmax(scores))
            best_var = candidates[best_local_idx]

            cand_features = features[candidates]  # Shape: (K, num_features)

            dataset.append(SearchStateSample(
                instance_name=inst.name,
                depth=depth,
                candidate_indices=candidates,
                candidate_features=cand_features,
                target_scores=scores,
                best_candidate_idx=best_local_idx,
            ))
            states_collected += 1

            # Follow best variable to continue trajectory
            values = csp.domain_values(best_var)
            for val in values:
                trail_mark = len(trail)
                current_mask = csp.domains[best_var]
                pruned = current_mask ^ (1 << val)
                csp.domains[best_var] = (1 << val)
                trail.append((best_var, pruned))

                arcs = [(k, best_var) for k in csp.neighbors[best_var]]
                prop_ok = ac3_propagate(csp, arcs=arcs, trail=trail, track_weights=False)

                if prop_ok:
                    done = search_collect(depth + 1)
                    if done:
                        return True

                while len(trail) > trail_mark:
                    u_v, u_m = trail.pop()
                    csp.domains[u_v] |= u_m

            return False

        search_collect(depth=0)

    return dataset


def save_dataset(samples: List[SearchStateSample], path: str) -> None:
    """Save dataset samples to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(samples, f)


def load_dataset(path: str) -> List[SearchStateSample]:
    """Load dataset samples from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
