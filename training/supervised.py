"""Supervised training pipeline supporting multi-seed learning and validation scoring."""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Sequence, Optional
from pathlib import Path
import numpy as np

from labels.dataset import SearchStateSample
from models.base import BaseModel
from models.lightgbm_model import LightGBMModel
from models.mlp_model import MLPModel


def prepare_training_data(samples: Sequence[SearchStateSample]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten SearchStateSample objects into (X, y) feature-target matrices.
    
    Each candidate in a state becomes a row in X with target score in y.
    """
    X_list = []
    y_list = []

    for s in samples:
        X_list.append(s.candidate_features)
        y_list.append(s.target_scores)

    if not X_list:
        return np.zeros((0, 12), dtype=np.float32), np.zeros(0, dtype=np.float32)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    return X, y


def evaluate_ranking_accuracy(
    model: BaseModel,
    val_samples: Sequence[SearchStateSample],
) -> Dict[str, float]:
    """
    Evaluate model decision accuracy against oracle labels on validation states.
    
    Returns:
        Dict with top1_accuracy (fraction of states matching oracle's top choice)
        and mean squared error.
    """
    if not val_samples:
        return {"top1_accuracy": 0.0, "mse": 0.0, "num_samples": 0}

    correct_top1 = 0
    total_squared_error = 0.0
    total_candidates = 0

    for s in val_samples:
        pred_scores = model.score(s.candidate_features)
        pred_top1 = int(np.argmax(pred_scores))

        if pred_top1 == s.best_candidate_idx:
            correct_top1 += 1

        total_squared_error += float(np.sum((pred_scores - s.target_scores) ** 2))
        total_candidates += len(s.target_scores)

    return {
        "top1_accuracy": round(float(correct_top1 / len(val_samples)), 4),
        "mse": round(float(total_squared_error / max(1, total_candidates)), 6),
        "num_samples": len(val_samples),
    }


def train_supervised_models(
    train_samples: Sequence[SearchStateSample],
    val_samples: Sequence[SearchStateSample],
    model_type: str = "lightgbm",
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    save_dir: Optional[str] = "results/models",
) -> List[Tuple[BaseModel, Dict[str, float]]]:
    """
    Train learned models across at least 5 seeds for statistical stability.
    
    Args:
        train_samples: Labeled training states.
        val_samples: Labeled validation states.
        model_type: 'lightgbm' or 'mlp'.
        seeds: List of random seeds (minimum 5).
        save_dir: Directory to save trained model artifacts.
        
    Returns:
        List of (model, val_metrics) pairs.
    """
    X_train, y_train = prepare_training_data(train_samples)
    trained: List[Tuple[BaseModel, Dict[str, float]]] = []

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        if model_type == "lightgbm":
            model = LightGBMModel(seed=seed, name=f"LGBM_seed{seed}")
            model.fit(X_train, y_train)
        elif model_type == "mlp":
            model = MLPModel(seed=seed, name=f"MLP_seed{seed}")
            model.fit(X_train, y_train, epochs=30, batch_size=32)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lightgbm' or 'mlp'.")

        metrics = evaluate_ranking_accuracy(model, val_samples)
        metrics["seed"] = seed
        metrics["model_type"] = model_type

        if save_dir:
            model_path = Path(save_dir) / f"{model_type}_s{seed}.pkl"
            if model_type == "lightgbm":
                model.save(str(model_path))
            else:
                model.save(str(model_path.with_suffix(".pt")))

        trained.append((model, metrics))

    return trained
