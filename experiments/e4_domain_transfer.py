"""Experiment E4: Domain Transfer Evaluation (Graph Colouring -> Sudoku & Cryptarithmetic).

Trains models solely on Graph Colouring instances and evaluates zero-shot transfer
to structurally distinct CSP domains: Sudoku (grid AllDiffs) and Cryptarithmetic
(arithmetic carry decomposition).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from solver.backtracking import BacktrackingSolver
from heuristics.variable_ordering import VARIABLE_HEURISTICS
from heuristics.value_ordering import VALUE_HEURISTICS
from generators.graph_coloring import generate_graph_coloring
from generators.sudoku import generate_sudoku, parse_sudoku_string, PUBLISHED_HARD_PUZZLES
from generators.cryptarithmetic import get_classic_cryptarithmetic
from labels.dataset import collect_training_trajectories
from training.supervised import train_supervised_models
from training.gnn_training import collect_graph_trajectories, train_gnn_models
from analysis.metrics import aggregate_experiment_metrics, compute_oracle_gap_closed
from analysis.significance import paired_wilcoxon_test, compute_multi_seed_statistics
from analysis.plots import plot_baseline_comparison, plot_dolan_more
from analysis.tables import generate_markdown_table


def run_e4_experiment(cfg: Dict[str, Any]) -> None:
    exp_name = cfg.get("experiment_name", "e4_domain_transfer")
    seeds = cfg.get("seeds", [42, 43, 44, 45, 46])
    time_limit = float(cfg.get("time_limit_sec", 10.0))
    prop_mode = cfg.get("propagation_mode", "mac")
    verify_sol = bool(cfg.get("verify_solutions", True))

    print(f"\n=================================================================")
    print(f"  RUNNING EXPERIMENT E4: DOMAIN TRANSFER (SEEDS: {seeds})")
    print(f"=================================================================\n")

    # 1. Build Training Set: SOLELY Graph Colouring
    train_inst = []
    ds_cfg = cfg.get("dataset", {}).get("train_instances", {})

    col_cfg = ds_cfg.get("graph_coloring", {})
    for i in range(col_cfg.get("count", 20)):
        s = col_cfg.get("start_seed", 9000) + i
        num_v = col_cfg.get("num_vertices", 30) + (i % 3) * 5  # Variation in sizes: 30, 35, 40
        train_inst.append(generate_graph_coloring(
            num_vertices=num_v,
            num_colors=col_cfg.get("num_colors", 3),
            avg_degree=col_cfg.get("avg_degree", 4.5),
            planted=True,
            seed=s,
            name=f"train_col_v{num_v}_{s}",
        ))

    print(f"Training distribution: {len(train_inst)} Graph Colouring instances only.")

    # 2. Build Test Set: Zero-Shot Target Domains (Sudoku & Cryptarithmetic)
    test_inst = []
    test_cfg = cfg.get("dataset", {}).get("test_instances", {})

    sud_cfg = test_cfg.get("sudoku_transfer", {})
    for i in range(sud_cfg.get("count", 6)):
        s = sud_cfg.get("start_seed", 9500) + i
        diff = ["easy", "medium", "hard"][i % 3]
        test_inst.append(generate_sudoku(seed=s, difficulty=diff, name=f"transfer_sudoku_{diff}_{s}"))

    # Include published hard benchmarks
    for pkey in ["ai_escargot", "golden_nugget"]:
        if pkey in PUBLISHED_HARD_PUZZLES:
            test_inst.append(parse_sudoku_string(PUBLISHED_HARD_PUZZLES[pkey], name=f"transfer_pub_{pkey}", difficulty="hard"))

    crypto_cfg = test_cfg.get("cryptarithmetic_transfer", {})
    for pkey in crypto_cfg.get("puzzles", ["send_more_money", "cross_roads_danger", "donald_gerald_robert", "base_ball_games"]):
        test_inst.append(get_classic_cryptarithmetic(pkey))

    print(f"Zero-Shot Target Domains Test set: {len(test_inst)} instances (Sudoku & Cryptarithmetic).")

    # 3. Supervised Trajectory Collection & Training (Only on Graph Colouring)
    tr_cfg = cfg.get("training", {})
    print("\nCollecting base training trajectories on Graph Colouring...")
    base_samples = collect_training_trajectories(
        instances=train_inst,
        oracle_type=tr_cfg.get("oracle_type", "lookahead"),
        max_states_per_instance=tr_cfg.get("max_states_per_instance", 15),
        lookahead_depth=tr_cfg.get("lookahead_depth", 1),
        top_k_candidates=tr_cfg.get("top_k_candidates", 3),
        seed=42,
    )

    print("Collecting bipartite graph trajectories on Graph Colouring...")
    graph_samples = collect_graph_trajectories(
        instances=train_inst,
        oracle_type=tr_cfg.get("oracle_type", "lookahead"),
        max_states_per_instance=tr_cfg.get("max_states_per_instance", 15),
        lookahead_depth=tr_cfg.get("lookahead_depth", 1),
        top_k_candidates=tr_cfg.get("top_k_candidates", 3),
        seed=42,
    )

    val_samples = base_samples[-30:]
    core_train_samples = base_samples[:-30]

    # Train models across 5 seeds
    print("Training LightGBM & MLP models across 5 seeds...")
    lgbm_models = train_supervised_models(core_train_samples, val_samples, model_type="lightgbm", seeds=seeds)
    mlp_models = train_supervised_models(core_train_samples, val_samples, model_type="mlp", seeds=seeds)

    print("Training Bipartite GNN models across 5 seeds...")
    gnn_models = train_gnn_models(
        train_samples=graph_samples,
        seeds=seeds,
        epochs=tr_cfg.get("epochs", 25),
        batch_size=tr_cfg.get("batch_size", 16),
        save_dir="results/models",
    )

    # 4. Zero-Shot Evaluation on Target Domains
    eval_cfg = cfg.get("evaluation", {})
    val_heuristic = VALUE_HEURISTICS["natural"]
    all_runs: List[Dict[str, Any]] = []

    print("\n--- Evaluating Baselines on Transfer Domains ---")
    for b_name in eval_cfg.get("baselines", ["mrv", "mrv_degree", "dom_wdeg"]):
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

    print("--- Evaluating Zero-Shot Learned Models across Seeds ---")
    model_groups = [
        ("lgbm_zero_shot", lgbm_models, 3),
        ("mlp_zero_shot", mlp_models, 3),
        ("gnn_zero_shot", gnn_models, 3),
    ]

    for name, m_list, top_k in model_groups:
        for model, meta in m_list:
            m_seed = meta.get("seed", 42)
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
    results_dir = Path("results")
    csv_path = results_dir / f"{exp_name}_results.csv"
    df_runs.to_csv(csv_path, index=False)
    print(f"\nSaved raw E4 test runs to: {csv_path}")

    # 5. Significance & Multi-seed Metrics
    df_summary = aggregate_experiment_metrics(df_runs, time_limit_sec=time_limit)

    dom_wdeg_runs = df_runs[df_runs["heuristic"] == "dom_wdeg"].sort_values("instance")
    baseline_bt = dom_wdeg_runs["backtracks"].values
    baseline_time = dom_wdeg_runs["wall_time_sec"].values

    mrv_runs = df_runs[df_runs["heuristic"] == "mrv"].sort_values("instance")
    mrv_mean_bt = float(mrv_runs["backtracks"].mean()) if len(mrv_runs) > 0 else 1.0
    dom_mean_bt = float(dom_wdeg_runs["backtracks"].mean())

    stat_records = []
    for h_name, group in df_runs.groupby("heuristic"):
        inst_means = group.groupby("instance").mean(numeric_only=True).reindex(dom_wdeg_runs["instance"]).fillna(0)
        h_bt = inst_means["backtracks"].values
        h_time = inst_means["wall_time_sec"].values

        wilcox_bt = paired_wilcoxon_test(h_bt, baseline_bt)
        wilcox_time = paired_wilcoxon_test(h_time, baseline_time)

        model_seeds_bt = group.groupby("model_seed")["backtracks"].mean().values if "model_seed" in group and group["model_seed"].notnull().any() else [float(group["backtracks"].mean())]
        seed_stats = compute_multi_seed_statistics(model_seeds_bt)

        h_mean_bt = float(group["backtracks"].mean())
        gap_closed = compute_oracle_gap_closed(
            model_backtracks=h_mean_bt,
            mrv_backtracks=mrv_mean_bt,
            oracle_backtracks=dom_mean_bt,
        )

        stat_records.append({
            "heuristic": h_name,
            "mean_backtracks": round(h_mean_bt, 2),
            "bt_95ci": f"[{seed_stats['ci_lower']:.2f}, {seed_stats['ci_upper']:.2f}]",
            "par2_sec": round(float(df_summary[df_summary['heuristic'] == h_name]['par2_sec'].values[0]), 4),
            "solve_rate_pct": round(float(df_summary[df_summary['heuristic'] == h_name]['solve_rate_pct'].values[0]), 1),
            "wilcoxon_p_bt_vs_domwdeg": round(wilcox_bt["p_value"], 4),
            "wilcoxon_p_time_vs_domwdeg": round(wilcox_time["p_value"], 4),
            "gap_closed_pct": round(gap_closed, 1),
        })

    df_stats = pd.DataFrame(stat_records).sort_values("par2_sec")

    json_path = results_dir / f"{exp_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_stats.to_dict(orient="records"), f, indent=2)

    md_table = generate_markdown_table(df_stats, title=f"Experiment E4 Domain Transfer Results (Graph Colouring -> Sudoku & Cryptarithmetic)")
    md_path = results_dir / f"{exp_name}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + md_table)

    # 6. Generate E4 plots
    try:
        plot_path_bar = results_dir / f"{exp_name}_comparison.png"
        plot_baseline_comparison(df_summary, output_path=str(plot_path_bar))
        print(f"Saved E4 comparison bar plot to: {plot_path_bar}")

        plot_path_dm = results_dir / f"{exp_name}_dolan_more.png"
        plot_dolan_more(df_runs, metric="wall_time_sec", output_path=str(plot_path_dm))
        print(f"Saved E4 Dolan-Moré performance profile to: {plot_path_dm}")
    except Exception as e:
        print(f"Plotting error: {e}")
