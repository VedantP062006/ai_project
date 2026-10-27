"""Visualization suite: Dolan-Moré performance profiles, backtrack distributions, and runtime comparisons."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_dolan_more(
    df: pd.DataFrame,
    metric: str = "wall_time_sec",
    max_tau: float = 10.0,
    output_path: Optional[str] = "results/dolan_more_profile.png",
) -> None:
    """
    Generate and save a Dolan-Moré performance profile.
    
    r_{p, s} = metric(p, s) / min_s' metric(p, s')
    rho_s(tau) = fraction of instances where r_{p, s} <= tau
    """
    pivot = df.groupby(["instance", "heuristic"])[metric].mean().unstack()
    heuristics = list(pivot.columns)

    # Replace timeouts/zeros with safe lower bound
    min_per_inst = pivot.min(axis=1).replace(0, 1e-6)
    ratio_df = pivot.divide(min_per_inst, axis=0)

    # Points on tau axis
    taus = np.linspace(1.0, max_tau, 200)
    plt.figure(figsize=(8, 5), dpi=300)

    for h in heuristics:
        ratios = ratio_df[h].dropna().values
        rho = [np.mean(ratios <= t) for t in taus]
        plt.plot(taus, rho, label=h, linewidth=2)

    plt.xlabel(f"Performance Ratio tau ({metric})", fontsize=11)
    plt.ylabel("P(r_{p, s} <= tau)", fontsize=11)
    plt.title(f"Dolan-More Performance Profile ({metric})", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def plot_baseline_comparison(
    summary_df: pd.DataFrame,
    output_path: Optional[str] = "results/baseline_comparison.png",
) -> None:
    """Generate side-by-side bar plots for Mean Backtracks and PAR2 runtime."""
    heuristics = summary_df["heuristic"].tolist()
    mean_backtracks = summary_df["mean_backtracks"].tolist()
    par2 = summary_df["par2_sec"].tolist()

    x = np.arange(len(heuristics))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    color1 = "#2b5c8f"
    color2 = "#d95f02"

    ax1.set_xlabel("Heuristic", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Mean Backtracks", color=color1, fontweight="bold", fontsize=11)
    bars1 = ax1.bar(x - width/2, mean_backtracks, width, label="Mean Backtracks", color=color1, alpha=0.85)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("PAR2 (sec)", color=color2, fontweight="bold", fontsize=11)
    bars2 = ax2.bar(x + width/2, par2, width, label="PAR2 (sec)", color=color2, alpha=0.85)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(heuristics, rotation=20, ha="right", fontsize=10)
    plt.title("Baseline Heuristics: Backtracks vs PAR2 Runtime", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()
