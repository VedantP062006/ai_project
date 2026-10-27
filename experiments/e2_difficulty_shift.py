"""Experiment E2: Difficulty Shift Evaluation (Easy/Medium -> Hard & Published Puzzles)."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from solver.backtracking import BacktrackingSolver
from heuristics.variable_ordering import VARIABLE_HEURISTICS
from heuristics.value_ordering import VALUE_HEURISTICS
from generators.sudoku import generate_sudoku, parse_sudoku_string, PUBLISHED_HARD_PUZZLES
from labels.dataset import collect_training_trajectories
from training.supervised import train_supervised_models
from training.dagger import train_dagger
from analysis.metrics import aggregate_experiment_metrics, compute_oracle_gap_closed
from analysis.significance import paired_wilcoxon_test, compute_multi_seed_statistics
from analysis.plots import plot_baseline_comparison, plot_dolan_more
from analysis.tables import generate_markdown_table


def run_e2_experiment(cfg: Dict[str, Any]) -> None:
    exp_name = cfg.get("experiment_name", "e2_difficulty_shift")
    seeds = cfg.get("seeds", [42, 43, 44, 45, 46])
    time_limit = float(cfg.get("time_limit_sec", 10.0))
    prop_mode = cfg.get("propagation_mode", "mac")
    verify_sol = bool(cfg.get("verify_solutions", True))

    print(f"\n=================================================================")
    print(f"  RUNNING EXPERIMENT E2: DIFFICULTY SHIFT (SEEDS: {seeds})")
    print(f"=================================================================\n")

    # 1. Build Training Set: Easy and Medium Sudokus
    train_inst = []
    ds_cfg = cfg.get("dataset", {}).get("train_instances", {})

    easy_cfg = ds_cfg.get("sudoku_easy", {})
    for i in range(easy_cfg.get("count", 15)):
        s = easy_cfg.get("start_seed", 3000) + i
        train_inst.append(generate_sudoku(seed=s, difficulty="easy", name=f"sudoku_easy_{s}"))

    med_cfg = ds_cfg.get("sudoku_medium", {})
    for i in range(med_cfg.get("count", 15)):
        s = med_cfg.get("start_seed", 4000) + i
        train_inst.append(generate_sudoku(seed=s, difficulty="medium", name=f"sudoku_med_{s}"))

    print(f"Training distribution: {len(train_inst)} Easy/Medium instances.")

    # 2. Build Test Set: Hard Sudokus + Published Hard Benchmarks
    test_inst = []
    test_cfg = cfg.get("dataset", {}).get("test_instances", {})

    hard_cfg = test_cfg.get("sudoku_hard", {})
    for i in range(hard_cfg.get("count", 10)):
        s = hard_cfg.get("start_seed", 5000) + i
        test_inst.append(generate_sudoku(seed=s, difficulty="hard", name=f"sudoku_hard_{s}"))

    for pkey in test_cfg.get("published_hard", ["ai_escargot", "golden_nugget", "easter_monster"]):
        if pkey in PUBLISHED_HARD_PUZZLES:
            test_inst.append(parse_sudoku_string(PUBLISHED_HARD_PUZZLES[pkey], name=f"pub_{pkey}", difficulty="hard"))

    print(f"Out-of-Distribution Test set: {len(test_inst)} Hard and Published benchmark puzzles.")

    # 3. Supervised & DAgger Training
    tr_cfg = cfg.get("training", {})
    print(f"Collecting base training trajectories on Easy/Medium instances...")
    base_samples = collect_training_trajectories(
        instances=train_inst,
        oracle_type=tr_cfg.get("oracle_type", "lookahead"),
        max_states_per_instance=tr_cfg.get("max_states_per_instance", 20),
        lookahead_depth=tr_cfg.get("lookahead_depth", 1),
        top_k_candidates=tr_cfg.get("top_k_candidates", 3),
        seed=42,
    )
    # Split small validation set from train
    val_inst = train_inst[-6:]
    val_samples = base_samples[-40:]
    core_train_samples = base_samples[:-40]

    # Train Supervised Models (5 seeds)
    print("Training Supervised LightGBM & MLP models...")
    lgbm_sup = train_supervised_models(core_train_samples, val_samples, model_type="lightgbm", seeds=seeds)
    mlp_sup = train_supervised_models(core_train_samples, val_samples, model_type="mlp", seeds=seeds)

    # Train DAgger Models (5 seeds)
    print("Training DAgger Active Learning LightGBM & MLP models...")
    lgbm_dag = []
    mlp_dag = []
    for s in seeds:
        m_lgbm, _ = train_dagger(
            train_instances=train_inst[:20],
            val_instances=val_inst,
            initial_samples=core_train_samples,
            model_type="lightgbm",
            num_iterations=tr_cfg.get("dagger_iterations", 3),
            top_k=tr_cfg.get("top_k_candidates", 3),
            seed=s,
            save_dir="results/models",
        )
        lgbm_dag.append((m_lgbm, {"seed": s}))

        m_mlp, _ = train_dagger(
            train_instances=train_inst[:20],
            val_instances=val_inst,
            initial_samples=core_train_samples,
            model_type="mlp",
            num_iterations=tr_cfg.get("dagger_iterations", 3),
            top_k=tr_cfg.get("top_k_candidates", 3),
            seed=s,
            save_dir="results/models",
        )
        mlp_dag.append((m_mlp, {"seed": s}))

    # 4. Evaluation on Out-of-Distribution Test Set
    eval_cfg = cfg.get("evaluation", {})
    val_heuristic = VALUE_HEURISTICS["natural"]
    all_runs: List[Dict[str, Any]] = []

    print("\n--- Evaluating Baselines on Hard Instances ---")
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

    print("--- Evaluating Supervised & DAgger Learned Models across Seeds ---")
    model_groups = [
        ("lgbm_supervised", lgbm_sup, 3),
        ("lgbm_dagger", lgbm_dag, 3),
        ("mlp_supervised", mlp_sup, 3),
        ("mlp_dagger", mlp_dag, 3),
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
    print(f"\nSaved raw E2 test runs to: {csv_path}")

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
            "wilcoxon_p_bt_vs_domwdeg": round(wilcox_bt["p_value"], 4),
            "wilcoxon_p_time_vs_domwdeg": round(wilcox_time["p_value"], 4),
            "gap_closed_pct": round(gap_closed, 1),
        })

    df_stats = pd.DataFrame(stat_records).sort_values("par2_sec")

    json_path = results_dir / f"{exp_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_stats.to_dict(orient="records"), f, indent=2)

    md_table = generate_markdown_table(df_stats, title=f"Experiment E2 Difficulty Shift Results & Generalisation")
    md_path = results_dir / f"{exp_name}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + md_table)

    # 6. Generate E2 plots
    try:
        plot_path_bar = results_dir / f"{exp_name}_comparison.png"
        plot_baseline_comparison(df_summary, output_path=str(plot_path_bar))
        print(f"Saved E2 comparison bar plot to: {plot_path_bar}")

        plot_path_dm = results_dir / f"{exp_name}_dolan_more.png"
        plot_dolan_more(df_runs, metric="wall_time_sec", output_path=str(plot_path_dm))
        print(f"Saved E2 Dolan-Moré performance profile to: {plot_path_dm}")
    except Exception as e:
        print(f"Plotting error: {e}")
