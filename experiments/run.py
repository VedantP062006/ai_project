"""Unified experiment execution engine.

Usage:
    python -m experiments.run --config configs/baselines.yaml
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import yaml
import pandas as pd

from solver.backtracking import BacktrackingSolver
from heuristics.variable_ordering import VARIABLE_HEURISTICS
from heuristics.value_ordering import VALUE_HEURISTICS
from generators.base import CSPInstance
from generators.sudoku import generate_sudoku
from generators.graph_coloring import generate_graph_coloring
from generators.cryptarithmetic import get_classic_cryptarithmetic
from analysis.metrics import aggregate_experiment_metrics
from analysis.plots import plot_baseline_comparison, plot_dolan_more
from analysis.tables import generate_markdown_table


def load_instances_from_config(cfg: Dict[str, Any]) -> List[CSPInstance]:
    """Generate or load instances according to the YAML configuration."""
    instances: List[CSPInstance] = []
    inst_cfg = cfg.get("instances", {})

    for group_name, params in inst_cfg.items():
        gtype = params.get("type")
        if gtype == "sudoku":
            diff = params.get("difficulty", "medium")
            count = params.get("count", 5)
            start_seed = params.get("start_seed", 100)
            for i in range(count):
                inst = generate_sudoku(
                    seed=start_seed + i,
                    difficulty=diff,
                    name=f"sudoku_{diff}_{start_seed + i}",
                )
                instances.append(inst)

        elif gtype == "graph_coloring":
            num_v = params.get("num_vertices", 30)
            num_k = params.get("num_colors", 3)
            avg_deg = params.get("avg_degree", 4.5)
            count = params.get("count", 5)
            start_seed = params.get("start_seed", 200)
            for i in range(count):
                inst = generate_graph_coloring(
                    num_vertices=num_v,
                    num_colors=num_k,
                    avg_degree=avg_deg,
                    planted=True,
                    seed=start_seed + i,
                    name=f"col_v{num_v}_k{num_k}_{start_seed + i}",
                )
                instances.append(inst)

        elif gtype == "cryptarithmetic":
            puzzles = params.get("puzzles", ["send_more_money"])
            for pkey in puzzles:
                inst = get_classic_cryptarithmetic(pkey)
                instances.append(inst)

    return instances


def run_experiment(config_path: str) -> None:
    """Execute the experiment defined by the YAML config."""
    print(f"Loading experiment configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "experiment")
    if exp_name == "e1_in_distribution":
        from experiments.e1_in_distribution import run_e1_experiment
        run_e1_experiment(cfg)
        return
    elif exp_name == "e2_difficulty_shift":
        from experiments.e2_difficulty_shift import run_e2_experiment
        run_e2_experiment(cfg)
        return
    elif exp_name == "e3_scale_shift":
        from experiments.e3_scale_shift import run_e3_experiment
        run_e3_experiment(cfg)
        return
    elif exp_name == "e4_domain_transfer":
        from experiments.e4_domain_transfer import run_e4_experiment
        run_e4_experiment(cfg)
        return
    elif exp_name == "e5_time_limited":
        from experiments.e5_time_limited import run_e5_experiment
        run_e5_experiment(cfg)
        return
    time_limit = float(cfg.get("time_limit_sec", 10.0))
    prop_mode = cfg.get("propagation_mode", "mac")
    verify_sol = bool(cfg.get("verify_solutions", True))
    heuristic_names = cfg.get("heuristics", ["mrv_degree"])
    val_heuristic_name = cfg.get("value_heuristic", "natural")
    exp_seed = int(cfg.get("seed", 42))

    val_heuristic = VALUE_HEURISTICS[val_heuristic_name]

    instances = load_instances_from_config(cfg)
    print(f"Loaded {len(instances)} problem instances.")

    results: List[Dict[str, Any]] = []

    for h_name in heuristic_names:
        if h_name not in VARIABLE_HEURISTICS:
            print(f"Warning: heuristic '{h_name}' not found. Skipping.")
            continue

        var_heuristic = VARIABLE_HEURISTICS[h_name]
        solver = BacktrackingSolver(
            variable_heuristic=var_heuristic,
            value_heuristic=val_heuristic,
            propagation_mode=prop_mode,
            time_limit_sec=time_limit,
            verify=verify_sol,
        )

        print(f"\n--- Running heuristic: {h_name.upper()} ---")
        for idx, inst_template in enumerate(instances):
            inst = inst_template.clone()
            sol, stats = solver.solve(inst.csp, seed=exp_seed)

            row = {
                "experiment": exp_name,
                "heuristic": h_name,
                "value_heuristic": val_heuristic_name,
                "instance": inst.name,
                "family": inst.family,
                "difficulty": inst.difficulty,
                "status": stats.status,
                "nodes": stats.nodes,
                "backtracks": stats.backtracks,
                "propagation_calls": stats.propagation_calls,
                "wall_time_sec": round(stats.wall_time_sec, 6),
                "inference_time_sec": round(stats.inference_time_sec, 6),
                "max_depth": stats.max_depth,
                "seed": exp_seed,
            }
            results.append(row)
            print(
                f"[{idx+1}/{len(instances)}] {inst.name:<25} | "
                f"Status: {stats.status:<9} | Nodes: {stats.nodes:<5} | "
                f"Backtracks: {stats.backtracks:<4} | Time: {stats.wall_time_sec:.4f}s"
            )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    csv_path = results_dir / f"{exp_name}_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved raw per-instance results to: {csv_path}")

    # Summary aggregations
    df_summary = aggregate_experiment_metrics(df_results, time_limit_sec=time_limit)
    json_path = results_dir / f"{exp_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_summary.to_dict(orient="records"), f, indent=2)
    print(f"Saved summary metrics to: {json_path}")

    md_table = generate_markdown_table(df_summary, title=f"Experiment Summary: {exp_name}")
    md_path = results_dir / f"{exp_name}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + md_table)

    # Generate baseline plots
    try:
        plot_path_bar = results_dir / f"{exp_name}_comparison.png"
        plot_baseline_comparison(df_summary, output_path=str(plot_path_bar))
        print(f"Saved baseline comparison bar plot to: {plot_path_bar}")

        plot_path_dm = results_dir / f"{exp_name}_dolan_more.png"
        plot_dolan_more(df_results, metric="wall_time_sec", output_path=str(plot_path_dm))
        print(f"Saved Dolan-More performance profile to: {plot_path_dm}")
    except Exception as e:
        print(f"Plotting error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible CSP experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration file.",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
