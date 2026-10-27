"""PyTorch Multi-Layer Perceptron policy for variable ordering."""

from __future__ import annotations
from typing import Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.base import BaseModel
from features.extractor import NUM_FEATURES


class MLPNetwork(nn.Module):
    """Feedforward network with LayerNorm and ReLU activations."""

    def __init__(self, in_features: int = NUM_FEATURES, hidden_1: int = 64, hidden_2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPModel(BaseModel):
    """
    Lightweight PyTorch MLP variable ordering policy.
    Evaluated with torch.no_grad() for fast CPU search guidance.
    """

    def __init__(
        self,
        name: str = "PyTorch_MLP",
        in_features: int = NUM_FEATURES,
        hidden_1: int = 64,
        hidden_2: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__(name=name)
        torch.manual_seed(seed)
        self.seed = seed
        self.network = MLPNetwork(in_features=in_features, hidden_1=hidden_1, hidden_2=hidden_2)
        self.network.eval()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """Train MLP policy using MSE loss on oracle scores."""
        torch.manual_seed(self.seed)
        self.network.train()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.network(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        self.network.eval()

    def score(self, candidate_features: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            x = torch.from_numpy(candidate_features).float()
            preds = self.network(x).cpu().numpy()
            return np.asarray(preds, dtype=np.float32)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.network.state_dict(),
            "seed": self.seed,
        }, p)

    @classmethod
    def load(cls, path: str) -> MLPModel:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(seed=checkpoint.get("seed", 42))
        model.network.load_state_dict(checkpoint["state_dict"])
        model.network.eval()
        return model
