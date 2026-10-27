"""DAgger (Dataset Aggregation) iterative imitation learning loop."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any, Callable
from pathlib import Path
import numpy as np

from solver.csp import CSP
from solver.propagation import ac3_propagate
from solver.backtracking import BacktrackingSolver
from features.extractor import FeatureExtractor
from labels.oracle import bounded_lookahead_oracle
from labels.dataset import SearchStateSample
from models.base import BaseModel
from models.lightgbm_model import LightGBMModel
from models.mlp_model import MLPModel
from training.supervised import prepare_training_data, evaluate_ranking_accuracy
from heuristics.variable_ordering import get_top_k_mrv_candidates
from generators.base import CSPInstance


def run_dagger_iteration(
    model: BaseModel,
    instances: Sequence[CSPInstance],
    top_k: Optional[int] = 3,
    lookahead_depth: int = 1,
    max_states_per_instance: int = 25,
    beta: float = 0.0,  # 0.0 = pure policy rollouts (DAgger), 1.0 = pure oracle rollouts
    seed: int = 42,
) -> List[SearchStateSample]:
    """
    Rollout the current learned policy on instances, query the oracle on all visited states,
    and return newly labeled state samples.
    """
    rng = np.random.default_rng(seed)
    extractor = FeatureExtractor()
    new_samples: List[SearchStateSample] = []

    for inst in instances:
        csp = inst.csp.clone()
        trail = []
        ok = ac3_propagate(csp, trail=trail)
        if not ok or csp.is_complete():
            continue

        states_collected = 0

        def dagger_search(depth: int) -> bool:
            nonlocal states_collected
            if csp.is_complete() or states_collected >= max_states_per_instance:
                return True

            features, candidate_mask = extractor.extract(csp, depth=depth)
            unassigned = np.where(candidate_mask)[0]
            if len(unassigned) == 0:
                return True

            if top_k is not None and len(unassigned) > top_k:
                candidates = get_top_k_mrv_candidates(csp, k=top_k)
            else:
                candidates = list(unassigned)

            # Query oracle on visited state
            oracle_scores = bounded_lookahead_oracle(csp, candidates, lookahead_depth=lookahead_depth)
            best_local_idx = int(np.argmax(oracle_scores))

            cand_feats = features[candidates]
            new_samples.append(SearchStateSample(
                instance_name=inst.name,
                depth=depth,
                candidate_indices=candidates,
                candidate_features=cand_feats,
                target_scores=oracle_scores,
                best_candidate_idx=best_local_idx,
            ))
            states_collected += 1

            # Decide which variable to branch on: policy or oracle (beta-mix)
            if rng.random() < beta:
                branch_var = candidates[best_local_idx]
            else:
                pred_scores = model.score(cand_feats)
                branch_var = candidates[int(np.argmax(pred_scores))]

            # Value loop
            values = csp.domain_values(branch_var)
            for val in values:
                trail_mark = len(trail)
                current_mask = csp.domains[branch_var]
                pruned = current_mask ^ (1 << val)
                csp.domains[branch_var] = (1 << val)
                trail.append((branch_var, pruned))

                arcs = [(k, branch_var) for k in csp.neighbors[branch_var]]
                prop_ok = ac3_propagate(csp, arcs=arcs, trail=trail, track_weights=False)

                if prop_ok:
                    done = dagger_search(depth + 1)
                    if done:
                        return True

                while len(trail) > trail_mark:
                    u_v, u_m = trail.pop()
                    csp.domains[u_v] |= u_m

            return False

        dagger_search(depth=0)

    return new_samples


def train_dagger(
    train_instances: Sequence[CSPInstance],
    val_instances: Sequence[CSPInstance],
    initial_samples: Sequence[SearchStateSample],
    model_type: str = "lightgbm",
    num_iterations: int = 4,
    beta_decay: float = 0.5,
    top_k: Optional[int] = 3,
    lookahead_depth: int = 1,
    seed: int = 42,
    save_dir: Optional[str] = "results/models",
) -> Tuple[BaseModel, List[Dict[str, Any]]]:
    """
    Run full DAgger active learning loop across iterations.
    
    Args:
        train_instances: Training set problem instances.
        val_instances: Validation set problem instances.
        initial_samples: Labeled seed dataset (D_0 from oracle).
        model_type: 'lightgbm' or 'mlp'.
        num_iterations: Number of policy rollout & dataset aggregation rounds.
        beta_decay: Decay factor for expert mixing rate beta.
        top_k: Restrict candidates to top-k MRV candidates.
        lookahead_depth: Lookahead oracle depth.
        seed: Random seed.
        save_dir: Checkpoint directory.
        
    Returns:
        (best_trained_model, history_metrics)
    """
    history: List[Dict[str, Any]] = []
    aggregated_dataset: List[SearchStateSample] = list(initial_samples)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Train initial model on D_0
    X_train, y_train = prepare_training_data(aggregated_dataset)
    if model_type == "lightgbm":
        model: BaseModel = LightGBMModel(seed=seed, name=f"LGBM_DAgger_s{seed}")
        model.fit(X_train, y_train)
    else:
        model = MLPModel(seed=seed, name=f"MLP_DAgger_s{seed}")
        model.fit(X_train, y_train, epochs=25, batch_size=32)

    # Val evaluation
    val_samples = run_dagger_iteration(
        model=model,
        instances=val_instances,
        top_k=top_k,
        lookahead_depth=lookahead_depth,
        max_states_per_instance=15,
        beta=1.0,
        seed=seed,
    )
    val_metrics = evaluate_ranking_accuracy(model, val_samples)
    val_metrics["iteration"] = 0
    val_metrics["dataset_size"] = len(aggregated_dataset)
    history.append(val_metrics)

    # 2. DAgger iterations
    beta = 1.0
    for it in range(1, num_iterations + 1):
        beta *= beta_decay

        # Rollout policy and query oracle on newly visited states
        new_samples = run_dagger_iteration(
            model=model,
            instances=train_instances,
            top_k=top_k,
            lookahead_depth=lookahead_depth,
            max_states_per_instance=20,
            beta=beta,
            seed=seed + it,
        )

        aggregated_dataset.extend(new_samples)

        # Retrain model on aggregated dataset D
        X_train, y_train = prepare_training_data(aggregated_dataset)
        if model_type == "lightgbm":
            model = LightGBMModel(seed=seed, name=f"LGBM_DAgger_s{seed}")
            model.fit(X_train, y_train)
        else:
            model = MLPModel(seed=seed, name=f"MLP_DAgger_s{seed}")
            model.fit(X_train, y_train, epochs=25, batch_size=32)

        metrics = evaluate_ranking_accuracy(model, val_samples)
        metrics["iteration"] = it
        metrics["dataset_size"] = len(aggregated_dataset)
        history.append(metrics)

    if save_dir:
        save_path = Path(save_dir) / f"{model_type}_dagger_s{seed}.pkl"
        if model_type == "lightgbm":
            model.save(str(save_path))
        else:
            model.save(str(save_path.with_suffix(".pt")))

    return model, history
