"""LightGBM tabular ranking policy for ultra-fast CPU inference."""

from __future__ import annotations
from typing import Optional, Any
from pathlib import Path
import pickle
import numpy as np
import lightgbm as lgb

from models.base import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM gradient-boosted decision tree ranker for variable ordering.
    Optimized for microsecond-level CPU inference latency.
    """

    def __init__(
        self,
        name: str = "LightGBM",
        n_estimators: int = 50,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model: Optional[lgb.LGBMRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train LightGBM regressor on candidate features and oracle target scores.
        
        Args:
            X: np.ndarray of shape (N_samples, num_features).
            y: np.ndarray of shape (N_samples,).
        """
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(X, y)

    def score(self, candidate_features: np.ndarray) -> np.ndarray:
        if self.model is None:
            # Untrained fallback: uniform scores
            return np.ones(len(candidate_features), dtype=np.float32)

        preds = self.model.predict(candidate_features)
        return np.asarray(preds, dtype=np.float32)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> LightGBMModel:
        with open(path, "rb") as f:
            return pickle.load(f)
