"""Experiment E1: In-distribution learning and evaluation."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from solver.backtracking import BacktrackingSolver
from heuristics.variable_ordering import VARIABLE_HEURISTICS
from heuristics.value_ordering import VALUE_HEURISTICS
from generators.sudoku import generate_sudoku
from generators.graph_coloring import generate_graph_coloring
from generators.split import split_instances
from labels.dataset import collect_training_trajectories
from training.supervised import train_supervised_models
from analysis.metrics import aggregate_experiment_metrics, compute_oracle_gap_closed
from analysis.significance import paired_wilcoxon_test, compute_multi_seed_statistics
from analysis.plots import plot_baseline_comparison, plot_dolan_more
from analysis.tables import generate_markdown_table


def run_e1_experiment(cfg: Dict[str, Any]) -> None:
    exp_name = cfg.get("experiment_name", "e1_in_distribution")
    seeds = cfg.get("seeds", [42, 43, 44, 45, 46])
    time_limit = float(cfg.get("time_limit_sec", 10.0))
    prop_mode = cfg.get("propagation_mode", "mac")
    verify_sol = bool(cfg.get("verify_solutions", True))

    print(f"\n=================================================================")
    print(f"  RUNNING EXPERIMENT E1: IN-DISTRIBUTION (SEEDS: {seeds})")
    print(f"=================================================================\n")

    # 1. Generate in-distribution instance pool
    ds_cfg = cfg.get("dataset", {})
    inst_pool = []

    sudoku_cfg = ds_cfg.get("instances", {}).get("sudoku_medium", {})
    for i in range(sudoku_cfg.get("count", 24)):
        s = sudoku_cfg.get("start_seed", 1000) + i
        inst_pool.append(generate_sudoku(seed=s, difficulty="medium", name=f"sudoku_med_{s}"))

    col_cfg = ds_cfg.get("instances", {}).get("graph_coloring_medium", {})
    for i in range(col_cfg.get("count", 24)):
        s = col_cfg.get("start_seed", 2000) + i
        inst_pool.append(generate_graph_coloring(
            num_vertices=col_cfg.get("num_vertices", 30),
            num_colors=col_cfg.get("num_colors", 3),
            avg_degree=col_cfg.get("avg_degree", 4.4),
            planted=True,
            seed=s,
            name=f"col_v30_{s}",
        ))

    print(f"Generated {len(inst_pool)} total in-distribution instances.")

    # 2. Strict instance-level split
    train_inst, val_inst, test_inst = split_instances(
        inst_pool,
        train_ratio=ds_cfg.get("train_ratio", 0.70),
        val_ratio=ds_cfg.get("val_ratio", 0.15),
        test_ratio=ds_cfg.get("test_ratio", 0.15),
        seed=42,
    )
    print(f"Strict instance splits -> Train: {len(train_inst)}, Val: {len(val_inst)}, Test: {len(test_inst)} (0% node leakage).")

    # 3. Collect supervision labels on Train split
    tr_cfg = cfg.get("training", {})
    print(f"Collecting oracle search trajectories on {len(train_inst)} training instances...")
    train_samples = collect_training_trajectories(
        instances=train_inst,
        oracle_type=tr_cfg.get("oracle_type", "lookahead"),
        max_states_per_instance=tr_cfg.get("max_states_per_instance", 20),
        lookahead_depth=tr_cfg.get("lookahead_depth", 1),
        top_k_candidates=tr_cfg.get("top_k_oracle", 4),
        seed=42,
    )
    val_samples = collect_training_trajectories(
        instances=val_inst,
        oracle_type=tr_cfg.get("oracle_type", "lookahead"),
        max_states_per_instance=10,
        lookahead_depth=1,
        top_k_candidates=4,
        seed=42,
    )
    print(f"Collected {len(train_samples)} training state samples and {len(val_samples)} validation samples.")

    # 4. Multi-seed training of LightGBM and MLP models
    print(f"\nTraining LightGBM models across {len(seeds)} seeds...")
    lgbm_models = train_supervised_models(
        train_samples,
        val_samples,
        model_type="lightgbm",
        seeds=seeds,
        save_dir="results/models",
    )

    print(f"Training PyTorch MLP models across {len(seeds)} seeds...")
    mlp_models = train_supervised_models(
        train_samples,
        val_samples,
        model_type="mlp",
        seeds=seeds,
        save_dir="results/models",
    )

    # 5. Evaluate on held-out Test split
    eval_cfg = cfg.get("evaluation", {})
    baseline_names = eval_cfg.get("baselines", ["mrv", "mrv_degree", "dom_wdeg"])
    learned_cfgs = eval_cfg.get("learned_models", [])

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_runs: List[Dict[str, Any]] = []

    print(f"\n--- Evaluating Baselines on {len(test_inst)} Held-Out Test Instances ---")
    val_heuristic = VALUE_HEURISTICS["natural"]

    for b_name in baseline_names:
        var_h = VARIABLE_HEURISTICS[b_name]
        solver = BacktrackingSolver(
            variable_heuristic=var_h,
            value_heuristic=val_heuristic,
            propagation_mode=prop_mode,
            time_limit_sec=time_limit,
            verify=verify_sol,
        )
        for inst in test_inst:
            sol, stats = solver.solve(inst.csp.clone(), seed=42)
            all_runs.append({
                "experiment": exp_name,
                "heuristic": b_name,
                "model_seed": None,
                "instance": inst.name,
                "family": inst.family,
                "difficulty": inst.difficulty,
                "status": stats.status,
                "nodes": stats.nodes,
                "backtracks": stats.backtracks,
                "propagation_calls": stats.propagation_calls,
                "wall_time_sec": round(stats.wall_time_sec, 6),
                "inference_time_sec": 0.0,
            })

    print(f"--- Evaluating Learned Models across {len(seeds)} Training Seeds ---")
    for learned_spec in learned_cfgs:
        name = learned_spec["name"]
        mtype = learned_spec["type"]
        top_k = learned_spec.get("top_k")
        model_list = lgbm_models if mtype == "lightgbm" else mlp_models

        for model, _ in model_list:
            m_seed = model.seed
            h_fn = model.create_heuristic_fn(top_k=top_k)
            solver = BacktrackingSolver(
                variable_heuristic=h_fn,
                value_heuristic=val_heuristic,
                propagation_mode=prop_mode,
                time_limit_sec=time_limit,
                verify=verify_sol,
            )

            for inst in test_inst:
                sol, stats = solver.solve(inst.csp.clone(), seed=42)
                all_runs.append({
                    "experiment": exp_name,
                    "heuristic": name,
                    "model_seed": m_seed,
                    "instance": inst.name,
                    "family": inst.family,
                    "difficulty": inst.difficulty,
                    "status": stats.status,
                    "nodes": stats.nodes,
                    "backtracks": stats.backtracks,
                    "propagation_calls": stats.propagation_calls,
                    "wall_time_sec": round(stats.wall_time_sec, 6),
                    "inference_time_sec": round(stats.inference_time_sec, 6),
                })

    df_runs = pd.DataFrame(all_runs)
    csv_path = results_dir / f"{exp_name}_results.csv"
    df_runs.to_csv(csv_path, index=False)
    print(f"\nSaved raw test runs to: {csv_path}")

    # 6. Statistical Significance & Gap Closed Analysis
    df_summary = aggregate_experiment_metrics(df_runs, time_limit_sec=time_limit)

    # Paired Wilcoxon signed-rank tests against dom/wdeg
    dom_wdeg_runs = df_runs[df_runs["heuristic"] == "dom_wdeg"].sort_values("instance")
    baseline_bt = dom_wdeg_runs["backtracks"].values
    baseline_time = dom_wdeg_runs["wall_time_sec"].values

    mrv_runs = df_runs[df_runs["heuristic"] == "mrv"].sort_values("instance")
    mrv_mean_bt = float(mrv_runs["backtracks"].mean()) if len(mrv_runs) > 0 else 1.0

    stat_records = []
    for h_name, group in df_runs.groupby("heuristic"):
        # Aggregate across seeds per instance
        inst_means = group.groupby("instance").mean(numeric_only=True).reindex(dom_wdeg_runs["instance"]).fillna(0)
        h_bt = inst_means["backtracks"].values
        h_time = inst_means["wall_time_sec"].values

        wilcox_bt = paired_wilcoxon_test(h_bt, baseline_bt)
        wilcox_time = paired_wilcoxon_test(h_time, baseline_time)

        # Multi-seed stats if learned model
        model_seeds_bt = group.groupby("model_seed")["backtracks"].mean().values if "model_seed" in group and group["model_seed"].notnull().any() else [float(group["backtracks"].mean())]
        seed_stats = compute_multi_seed_statistics(model_seeds_bt)

        # Oracle gap closed
        h_mean_bt = float(group["backtracks"].mean())
        gap_closed = compute_oracle_gap_closed(
            model_backtracks=h_mean_bt,
            mrv_backtracks=mrv_mean_bt,
            oracle_backtracks=float(dom_wdeg_runs["backtracks"].mean()),
        )

        stat_records.append({
            "heuristic": h_name,
            "mean_backtracks": round(h_mean_bt, 2),
            "bt_95ci": f"[{seed_stats['ci_lower']:.2f}, {seed_stats['ci_upper']:.2f}]",
            "par2_sec": round(float(df_summary[df_summary['heuristic'] == h_name]['par2_sec'].values[0]), 4),
            "wilcoxon_p_bt_vs_domwdeg": round(wilcox_bt["p_value"], 4),
            "wilcoxon_p_time_vs_domwdeg": round(wilcox_time["p_value"], 4),
            "gap_closed_pct": round(gap_closed, 1),
        })

    df_stats = pd.DataFrame(stat_records).sort_values("par2_sec")

    json_path = results_dir / f"{exp_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_stats.to_dict(orient="records"), f, indent=2)

    md_table = generate_markdown_table(df_stats, title=f"Experiment E1 In-Distribution Results & Significance")
    md_path = results_dir / f"{exp_name}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + md_table)

    # 7. Generate E1 plots
    try:
        plot_path_bar = results_dir / f"{exp_name}_comparison.png"
        plot_baseline_comparison(df_summary, output_path=str(plot_path_bar))
        print(f"Saved E1 comparison bar plot to: {plot_path_bar}")

        plot_path_dm = results_dir / f"{exp_name}_dolan_more.png"
        plot_dolan_more(df_runs, metric="wall_time_sec", output_path=str(plot_path_dm))
        print(f"Saved E1 Dolan-Moré performance profile to: {plot_path_dm}")
    except Exception as e:
        print(f"Plotting error: {e}")
