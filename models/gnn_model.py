"""Bipartite Graph Neural Network (GNN) policy for CSP variable ordering.

Constructs a bipartite graph between Variable nodes and Constraint nodes,
executing message passing over incident scopes to learn permutation- and scale-invariant
structural variable orderings.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any, Callable
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from solver.csp import CSP
from solver.constraints import BinaryNotEqualConstraint, BinaryTableConstraint, AllDifferentConstraint
from features.extractor import FeatureExtractor, NUM_FEATURES
from heuristics.variable_ordering import get_top_k_mrv_candidates
from models.base import BaseModel


VAR_FEAT_DIM = NUM_FEATURES  # 12 scale-invariant dynamic features
CON_FEAT_DIM = 7             # Arity, log_weight, assigned_frac, min_dom, 3x one-hot type


@dataclass
class GraphSearchStateSample:
    """A labeled bipartite graph decision state sampled during search."""
    instance_name: str
    depth: int
    candidate_indices: List[int]
    var_features: np.ndarray      # Shape: (num_vars, 12)
    con_features: np.ndarray      # Shape: (num_cons, 7)
    edge_var_indices: np.ndarray  # Shape: (num_edges,)
    edge_con_indices: np.ndarray  # Shape: (num_edges,)
    target_scores: np.ndarray     # Shape: (len(candidates),)
    best_candidate_idx: int


def build_bipartite_graph(
    csp: CSP,
    depth: int = 0,
    extractor: Optional[FeatureExtractor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Extract bipartite Variable-Constraint graph representation from a CSP state.

    Returns:
        var_feats: Tensor of shape (num_variables, VAR_FEAT_DIM)
        con_feats: Tensor of shape (num_constraints, CON_FEAT_DIM)
        edge_var: LongTensor of variable indices in edges (num_edges,)
        edge_con: LongTensor of constraint indices in edges (num_edges,)
        candidate_mask: Boolean array of unassigned variables
    """
    if extractor is None:
        extractor = FeatureExtractor()

    # 1. Variable features
    var_feats_np, candidate_mask = extractor.extract(csp, depth=depth)
    num_vars = csp.num_variables

    # 2. Constraint features and bipartite edges
    num_cons = len(csp.constraints)
    if num_cons == 0:
        # Dummy constraint connecting to all variables to maintain valid tensor dimensions
        con_feats_np = np.zeros((1, CON_FEAT_DIM), dtype=np.float32)
        edge_var_list = list(range(num_vars))
        edge_con_list = [0] * num_vars
    else:
        con_feats_np = np.zeros((num_cons, CON_FEAT_DIM), dtype=np.float32)
        edge_var_list: List[int] = []
        edge_con_list: List[int] = []

        max_vars_f = max(1.0, float(num_vars))
        for c_idx, c in enumerate(csp.constraints):
            scope = c.scope
            arity = len(scope)
            arity_norm = arity / max_vars_f
            log_wt = np.log1p(float(c.weight))

            # Scope assigned fraction
            assigned_count = sum(1 for v in scope if (csp.domains[v] & (csp.domains[v] - 1)) == 0 and csp.domains[v] != 0)
            assigned_frac = assigned_count / max(1.0, float(arity))

            # Minimum domain among unassigned in scope
            unassigned_doms = [csp.domain_size(v) for v in scope if not csp.is_assigned(v)]
            min_dom = (min(unassigned_doms) / 9.0) if unassigned_doms else 0.0

            # Constraint type one-hot (3 classes: BinaryNotEqual, BinaryTable, AllDifferent)
            type_vec = [0.0, 0.0, 0.0]
            if isinstance(c, BinaryNotEqualConstraint):
                type_vec[0] = 1.0
            elif isinstance(c, BinaryTableConstraint):
                type_vec[1] = 1.0
            elif isinstance(c, AllDifferentConstraint):
                type_vec[2] = 1.0

            con_feats_np[c_idx, 0] = arity_norm
            con_feats_np[c_idx, 1] = log_wt
            con_feats_np[c_idx, 2] = assigned_frac
            con_feats_np[c_idx, 3] = min_dom
            con_feats_np[c_idx, 4:7] = type_vec

            for v in scope:
                edge_var_list.append(v)
                edge_con_list.append(c_idx)

    var_feats = torch.from_numpy(var_feats_np).float()
    con_feats = torch.from_numpy(con_feats_np).float()
    edge_var = torch.tensor(edge_var_list, dtype=torch.long)
    edge_con = torch.tensor(edge_con_list, dtype=torch.long)

    return var_feats, con_feats, edge_var, edge_con, candidate_mask


class BipartiteGNNLayer(nn.Module):
    """
    Single bipartite message-passing layer between variables and constraints.
    Variables -> Constraints -> Variables.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # V -> C message & update
        self.msg_v2c = nn.Linear(hidden_dim, hidden_dim)
        self.con_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # C -> V message & update
        self.msg_c2v = nn.Linear(hidden_dim, hidden_dim)
        self.var_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        var_h: torch.Tensor,
        con_h: torch.Tensor,
        edge_var: torch.Tensor,
        edge_con: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_vars = var_h.size(0)
        num_cons = con_h.size(0)

        # 1. Variables -> Constraints
        m_v = self.msg_v2c(var_h[edge_var])
        con_agg = torch.zeros(num_cons, self.hidden_dim, device=var_h.device)
        con_agg.index_add_(0, edge_con, m_v)
        new_con_h = self.con_update(torch.cat([con_h, con_agg], dim=-1))

        # 2. Constraints -> Variables
        m_c = self.msg_c2v(new_con_h[edge_con])
        var_agg = torch.zeros(num_vars, self.hidden_dim, device=var_h.device)
        var_agg.index_add_(0, edge_var, m_c)
        new_var_h = self.var_update(torch.cat([var_h, var_agg], dim=-1))

        return new_var_h, new_con_h


class BipartiteGNNNetwork(nn.Module):
    """Bipartite Graph Neural Network for Variable Selection."""

    def __init__(
        self,
        in_var_dim: int = VAR_FEAT_DIM,
        in_con_dim: int = CON_FEAT_DIM,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.var_proj = nn.Sequential(
            nn.Linear(in_var_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.con_proj = nn.Sequential(
            nn.Linear(in_con_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            BipartiteGNNLayer(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        var_feats: torch.Tensor,
        con_feats: torch.Tensor,
        edge_var: torch.Tensor,
        edge_con: torch.Tensor,
    ) -> torch.Tensor:
        var_h = self.var_proj(var_feats)
        con_h = self.con_proj(con_feats)

        for layer in self.layers:
            var_h, con_h = layer(var_h, con_h, edge_var, edge_con)

        scores = self.readout(var_h).squeeze(-1)
        return scores


class GNNModel(BaseModel):
    """
    Bipartite Graph Neural Network variable-ordering policy.
    Evaluated with torch.no_grad() and isolated inference timing.
    """

    def __init__(
        self,
        name: str = "Bipartite_GNN",
        in_var_dim: int = VAR_FEAT_DIM,
        in_con_dim: int = CON_FEAT_DIM,
        hidden_dim: int = 32,
        num_layers: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__(name=name)
        torch.manual_seed(seed)
        self.seed = seed
        self.network = BipartiteGNNNetwork(
            in_var_dim=in_var_dim,
            in_con_dim=in_con_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.network.eval()
        self.extractor = FeatureExtractor()

    def fit_on_graphs(
        self,
        samples: Sequence[GraphSearchStateSample],
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 16,
    ) -> None:
        """Train GNN policy on collected bipartite graph search state samples."""
        if len(samples) == 0:
            return

        self.network.train()
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        criterion = nn.MSELoss()

        indices = np.arange(len(samples))
        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0

            for start in range(0, len(samples), batch_size):
                batch_idxs = indices[start : start + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0)

                for idx in batch_idxs:
                    s = samples[idx]
                    var_t = torch.from_numpy(s.var_features).float()
                    con_t = torch.from_numpy(s.con_features).float()
                    e_v = torch.from_numpy(s.edge_var_indices).long()
                    e_c = torch.from_numpy(s.edge_con_indices).long()
                    targets = torch.from_numpy(s.target_scores).float()

                    scores = self.network(var_t, con_t, e_v, e_c)
                    cand_preds = scores[s.candidate_indices]

                    loss = criterion(cand_preds, targets)
                    batch_loss = batch_loss + loss

                batch_loss = batch_loss / max(1, len(batch_idxs))
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

        self.network.eval()

    def score_graph(self, csp: CSP, depth: int = 0) -> np.ndarray:
        """Run GNN inference on current CSP search state, tracking inference latency."""
        t0 = time.perf_counter()
        var_t, con_t, e_v, e_c, _ = build_bipartite_graph(csp, depth=depth, extractor=self.extractor)
        with torch.no_grad():
            scores = self.network(var_t, con_t, e_v, e_c).cpu().numpy()
        self.last_inference_time = time.perf_counter() - t0
        return scores

    def score(self, candidate_features: np.ndarray) -> np.ndarray:
        """Fallback scoring using mean feature projection if graph not provided."""
        t0 = time.perf_counter()
        # Default fallback heuristic based on linear projection
        weights = np.array([-1.0, -1.0, 0.5, 0.5, -0.2, 1.0, -0.1, 0.0, 0.0, -0.5, -0.5, 0.0], dtype=np.float32)
        scores = np.dot(candidate_features, weights[: candidate_features.shape[1]])
        self.last_inference_time = time.perf_counter() - t0
        return scores

    def select_variable(
        self,
        csp: CSP,
        timer: Optional[Any] = None,
        top_k: Optional[int] = None,
        depth: int = 0,
    ) -> int:
        """Select next variable using GNN bipartite graph inference."""
        if top_k is not None:
            candidates = get_top_k_mrv_candidates(csp, k=top_k)
        else:
            candidates = [
                v for v in range(csp.num_variables)
                if not csp.is_assigned(v) and not csp.is_wiped_out(v)
            ]

        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidates[0]

        if timer is not None and hasattr(timer, "measure_inference"):
            with timer.measure_inference():
                scores = self.score_graph(csp, depth=depth)
        else:
            scores = self.score_graph(csp, depth=depth)

        cand_scores = [scores[c] for c in candidates]
        best_idx = int(np.argmax(cand_scores))
        return candidates[best_idx]

    def create_heuristic_fn(self, top_k: Optional[int] = 3) -> Callable[..., int]:
        """
        Create variable selection function for BacktrackingSolver.
        Accepts (csp, model, timer) to conform to solver's VariableOrderingFn signature.
        """
        model = self

        def heuristic(csp: CSP, _model: Any = None, timer: Any = None) -> int:
            return model.select_variable(csp, timer=timer, top_k=top_k)

        heuristic.__name__ = f"{self.name}_top{top_k}" if top_k else f"{self.name}_full"
        return heuristic

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "name": self.name,
            "seed": self.seed,
            "state_dict": self.network.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        data = torch.load(filepath, map_location="cpu")
        self.name = data.get("name", self.name)
        self.seed = data.get("seed", self.seed)
        self.network.load_state_dict(data["state_dict"])
        self.network.eval()
