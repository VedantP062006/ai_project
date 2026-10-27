"""Metric computation utilities: PAR2, solve rates, gap closure, and summary aggregations."""

from __future__ import annotations
from typing import Dict, List, Any, Sequence
import pandas as pd
import numpy as np


def compute_par2(df: pd.DataFrame, time_limit_sec: float = 10.0) -> float:
    """
    Compute Penalized Average Runtime 2 (PAR2).
    Solved instances are penalized by their actual wall-clock runtime.
    Timed out / unsolved instances are penalized by 2 * time_limit_sec.
    """
    if len(df) == 0:
        return 0.0
    runtimes = df["wall_time_sec"].copy()
    unsolved_mask = (df["status"] != "SOLVED")
    runtimes[unsolved_mask] = 2.0 * time_limit_sec
    return float(runtimes.mean())


def compute_oracle_gap_closed(
    model_backtracks: float,
    mrv_backtracks: float,
    oracle_backtracks: float,
) -> float:
    """
    Fraction of the gap between MRV and the lookahead oracle closed by the model:
    GapClosed = (MRV - Model) / (MRV - Oracle) * 100%
    """
    denom = mrv_backtracks - oracle_backtracks
    if abs(denom) < 1e-6:
        return 0.0
    return float((mrv_backtracks - model_backtracks) / denom * 100.0)


def aggregate_experiment_metrics(df: pd.DataFrame, time_limit_sec: float = 10.0) -> pd.DataFrame:
    """
    Aggregate per-heuristic metrics across instances:
    Solve Rate (%), Mean/Median Nodes, Mean/Median Backtracks, Mean/Median Runtime, PAR2.
    """
    records = []
    for heuristic, group in df.groupby("heuristic"):
        total = len(group)
        solved = (group["status"] == "SOLVED").sum()
        solve_rate = (solved / total) * 100.0 if total > 0 else 0.0
        par2 = compute_par2(group, time_limit_sec=time_limit_sec)

        records.append({
            "heuristic": heuristic,
            "instances": total,
            "solved": int(solved),
            "solve_rate_pct": round(solve_rate, 2),
            "mean_nodes": round(float(group["nodes"].mean()), 2),
            "median_nodes": round(float(group["nodes"].median()), 2),
            "mean_backtracks": round(float(group["backtracks"].mean()), 2),
            "median_backtracks": round(float(group["backtracks"].median()), 2),
            "mean_wall_time_sec": round(float(group["wall_time_sec"].mean()), 4),
            "median_wall_time_sec": round(float(group["wall_time_sec"].median()), 4),
            "par2_sec": round(par2, 4),
        })

    return pd.DataFrame(records).sort_values(by="par2_sec")
