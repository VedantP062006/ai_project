"""Trajectory collection and multi-seed training pipeline for Bipartite GNN policies."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
import numpy as np
import torch

from solver.csp import CSP
from solver.propagation import ac3_propagate
from heuristics.variable_ordering import get_top_k_mrv_candidates
from labels.oracle import bounded_lookahead_oracle, mrv_imitation_oracle
from models.gnn_model import GNNModel, GraphSearchStateSample, build_bipartite_graph
from generators.base import CSPInstance


def collect_graph_trajectories(
    instances: Sequence[CSPInstance],
    oracle_type: str = "lookahead",
    max_states_per_instance: int = 25,
    lookahead_depth: int = 1,
    top_k_candidates: Optional[int] = 3,
    seed: int = 42,
) -> List[GraphSearchStateSample]:
    """
    Collect labeled bipartite graph search states across problem instances.
    """
    rng = np.random.default_rng(seed)
    dataset: List[GraphSearchStateSample] = []

    for inst in instances:
        csp = inst.csp.clone()
        trail = []
        ok = ac3_propagate(csp, trail=trail)
        if not ok or csp.is_complete():
            continue

        states_collected = 0

        def sample_search(depth: int) -> bool:
            nonlocal states_collected
            if csp.is_complete() or states_collected >= max_states_per_instance:
                return True

            var_t, con_t, edge_var, edge_con, candidate_mask = build_bipartite_graph(csp, depth=depth)
            unassigned = np.where(candidate_mask)[0]
            if len(unassigned) == 0:
                return True

            if top_k_candidates is not None and len(unassigned) > top_k_candidates:
                candidates = get_top_k_mrv_candidates(csp, k=top_k_candidates)
            else:
                candidates = list(unassigned)

            if oracle_type == "mrv":
                oracle_scores = mrv_imitation_oracle(csp, candidates)
            else:
                oracle_scores = bounded_lookahead_oracle(csp, candidates, lookahead_depth=lookahead_depth)

            best_local_idx = int(np.argmax(oracle_scores))
            best_var = candidates[best_local_idx]

            dataset.append(GraphSearchStateSample(
                instance_name=inst.name,
                depth=depth,
                candidate_indices=candidates,
                var_features=var_t.numpy(),
                con_features=con_t.numpy(),
                edge_var_indices=edge_var.numpy(),
                edge_con_indices=edge_con.numpy(),
                target_scores=oracle_scores,
                best_candidate_idx=best_local_idx,
            ))
            states_collected += 1

            # Value loop with undo trail
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
                    done = sample_search(depth + 1)
                    if done:
                        return True

                while len(trail) > trail_mark:
                    u_v, u_m = trail.pop()
                    csp.domains[u_v] |= u_m

            return False

        sample_search(depth=0)

    return dataset


def train_gnn_models(
    train_samples: Sequence[GraphSearchStateSample],
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 16,
    save_dir: Optional[str] = "results/models",
) -> List[Tuple[GNNModel, Dict[str, Any]]]:
    """
    Train Bipartite GNN models across multiple independent random seeds.
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    trained_models: List[Tuple[GNNModel, Dict[str, Any]]] = []

    for seed in seeds:
        model = GNNModel(seed=seed, name=f"GNN_s{seed}")
        model.fit_on_graphs(train_samples, epochs=epochs, lr=lr, batch_size=batch_size)

        if save_dir:
            ckpt_path = Path(save_dir) / f"gnn_s{seed}.pt"
            model.save(str(ckpt_path))

        trained_models.append((model, {"seed": seed}))

    return trained_models
