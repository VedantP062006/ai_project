"""Abstract base policy interface for learned variable ordering."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Any
import numpy as np

from solver.csp import CSP
from features.extractor import FeatureExtractor
from heuristics.variable_ordering import get_top_k_mrv_candidates


class BaseModel(ABC):
    """Abstract base class for all learned variable ordering models."""

    def __init__(self, name: str = "LearnedModel") -> None:
        self.name = name
        self._extractor = FeatureExtractor()

    @abstractmethod
    def score(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Score candidate variables given their extracted features.
        
        Args:
            candidate_features: np.ndarray of shape (K, num_features).
            
        Returns:
            scores: np.ndarray of shape (K,), where higher means higher priority.
        """
        pass

    def select_variable(
        self,
        csp: CSP,
        timer: Optional[Any] = None,
        top_k: Optional[int] = None,
        depth: int = 0,
    ) -> int:
        """
        Select the next unassigned variable to branch on.
        
        Args:
            csp: Current CSP state.
            timer: Optional InstrumentationTimer to track inference time separately.
            top_k: If set, restricts candidate pool to top-k MRV candidates (H4).
            depth: Current search tree depth.
            
        Returns:
            Selected variable index, or -1 if all are assigned.
        """
        if top_k is not None:
            candidates = get_top_k_mrv_candidates(csp, k=top_k)
        else:
            candidates = [v for v in range(csp.num_variables) if not csp.is_assigned(v)]

        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidates[0]

        # Extract features for current state
        features, _ = self._extractor.extract(csp, depth=depth)
        candidate_features = features[candidates]

        # Evaluate model with isolated inference timing
        if timer is not None and hasattr(timer, "measure_inference"):
            with timer.measure_inference():
                scores = self.score(candidate_features)
        else:
            scores = self.score(candidate_features)

        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    def create_heuristic_fn(self, top_k: Optional[int] = None):
        """Create a callable matching VariableOrderingFn signature."""
        def heuristic(csp: CSP, model: Any = None, timer: Any = None) -> int:
            return self.select_variable(csp, timer=timer, top_k=top_k)
        return heuristic
