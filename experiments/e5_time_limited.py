"""Experiment E5: Time-Limited Benchmark with PAR2 and Solve Rate Evaluation."""

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
from generators.graph_coloring import generate_graph_coloring
from labels.dataset import collect_training_trajectories
from training.supervised import train_supervised_models
from training.dagger import train_dagger
from analysis.metrics import aggregate_experiment_metrics, compute_par2
from analysis.significance import paired_wilcoxon_test, compute_multi_seed_statistics
from analysis.plots import plot_baseline_comparison, plot_dolan_more
from analysis.tables import generate_markdown_table


def run_e5_experiment(cfg: Dict[str, Any]) -> None:
    exp_name = cfg.get("experiment_name", "e5_time_limited")
    seeds = cfg.get("seeds", [42, 43, 44, 45, 46])
    time_limit = float(cfg.get("time_limit_sec", 5.0))
    prop_mode = cfg.get("propagation_mode", "mac")
    verify_sol = bool(cfg.get("verify_solutions", True))

    print(f"\n=================================================================")
    print(f"  RUNNING EXPERIMENT E5: TIME-LIMITED (TIMEOUT: {time_limit}s, SEEDS: {seeds})")
    print(f"=================================================================\n")

    # 1. Load Hard instances (Sudoku Hard + Published Hard + Graph Coloring Hard)
    test_inst = []
    ds_cfg = cfg.get("dataset", {}).get("instances", {})

    hard_sudoku_cfg = ds_cfg.get("sudoku_hard", {})
    for i in range(hard_sudoku_cfg.get("count", 8)):
        s = hard_sudoku_cfg.get("start_seed", 6000) + i
        test_inst.append(generate_sudoku(seed=s, difficulty="hard", name=f"sudoku_hard_{s}"))

    for pkey in ds_cfg.get("published_hard", ["ai_escargot", "golden_nugget", "easter_monster"]):
        if pkey in PUBLISHED_HARD_PUZZLES:
            test_inst.append(parse_sudoku_string(PUBLISHED_HARD_PUZZLES[pkey], name=f"pub_{pkey}", difficulty="hard"))

    hard_col_cfg = ds_cfg.get("graph_coloring_hard", {})
    for i in range(hard_col_cfg.get("count", 8)):
        s = hard_col_cfg.get("start_seed", 7000) + i
        test_inst.append(generate_graph_coloring(
            num_vertices=hard_col_cfg.get("num_vertices", 40),
            num_colors=hard_col_cfg.get("num_colors", 3),
            avg_degree=hard_col_cfg.get("avg_degree", 4.6),
            planted=True,
            seed=s,
            name=f"col_v40_{s}",
        ))

    print(f"Challenging test benchmark set: {len(test_inst)} instances under {time_limit}s limit.")

    # 2. Build / Load Training Set
    tr_cfg = cfg.get("training", {})
    train_inst_cfg = tr_cfg.get("train_instances", {})
    train_inst = []

    sud_tr = train_inst_cfg.get("sudoku", {})
    for i in range(sud_tr.get("count", 15)):
        s = sud_tr.get("start_seed", 8000) + i
        train_inst.append(generate_sudoku(seed=s, difficulty=sud_tr.get("difficulty", "medium"), name=f"tr_sudoku_{s}"))

    col_tr = train_inst_cfg.get("graph_coloring", {})
    for i in range(col_tr.get("count", 15)):
        s = col_tr.get("start_seed", 9000) + i
        train_inst.append(generate_graph_coloring(
            num_vertices=col_tr.get("num_vertices", 25),
            num_colors=3,
            avg_degree=4.3,
            planted=True,
            seed=s,
            name=f"tr_col_{s}",
        ))

    print(f"Collecting training trajectories on {len(train_inst)} instances...")
    base_samples = collect_training_trajectories(
        instances=train_inst,
        oracle_type=tr_cfg.get("oracle_type", "lookahead"),
        max_states_per_instance=tr_cfg.get("max_states_per_instance", 20),
        lookahead_depth=tr_cfg.get("lookahead_depth", 1),
        top_k_candidates=tr_cfg.get("top_k_candidates", 3),
        seed=42,
    )
    val_samples = base_samples[-30:]
    core_samples = base_samples[:-30]

    # Train Supervised & DAgger models across seeds
    print("Training Supervised Models across seeds...")
    lgbm_sup = train_supervised_models(core_samples, val_samples, model_type="lightgbm", seeds=seeds)
    mlp_sup = train_supervised_models(core_samples, val_samples, model_type="mlp", seeds=seeds)

    print("Training DAgger Models across seeds...")
    lgbm_dag = []
    mlp_dag = []
    for s in seeds:
        m_lgbm, _ = train_dagger(
            train_instances=train_inst[:15],
            val_instances=train_inst[-5:],
            initial_samples=core_samples,
            model_type="lightgbm",
            num_iterations=tr_cfg.get("dagger_iterations", 3),
            top_k=tr_cfg.get("top_k_candidates", 3),
            seed=s,
            save_dir="results/models",
        )
        lgbm_dag.append((m_lgbm, {"seed": s}))

        m_mlp, _ = train_dagger(
            train_instances=train_inst[:15],
            val_instances=train_inst[-5:],
            initial_samples=core_samples,
            model_type="mlp",
            num_iterations=tr_cfg.get("dagger_iterations", 3),
            top_k=tr_cfg.get("top_k_candidates", 3),
            seed=s,
            save_dir="results/models",
        )
        mlp_dag.append((m_mlp, {"seed": s}))

    # 3. Evaluate under Strict Time Limit
    eval_cfg = cfg.get("evaluation", {})
    val_heuristic = VALUE_HEURISTICS["natural"]
    all_runs: List[Dict[str, Any]] = []

    print(f"\n--- Evaluating Baselines under {time_limit}s Limit ---")
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

    print(f"--- Evaluating Learned Models under {time_limit}s Limit ---")
    model_groups = [
        ("lgbm_top3", lgbm_sup, 3),
        ("lgbm_dagger_top3", lgbm_dag, 3),
        ("mlp_top3", mlp_sup, 3),
        ("mlp_dagger_top3", mlp_dag, 3),
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
    print(f"\nSaved raw E5 test runs to: {csv_path}")

    # 4. Significance & PAR2 Aggregations
    df_summary = aggregate_experiment_metrics(df_runs, time_limit_sec=time_limit)

    dom_wdeg_runs = df_runs[df_runs["heuristic"] == "dom_wdeg"].sort_values("instance")
    baseline_time = dom_wdeg_runs["wall_time_sec"].values
    baseline_bt = dom_wdeg_runs["backtracks"].values

    stat_records = []
    for h_name, group in df_runs.groupby("heuristic"):
        total_runs = len(group)
        solved_runs = (group["status"] == "SOLVED").sum()
        solve_rate = (solved_runs / total_runs) * 100.0 if total_runs > 0 else 0.0

        inst_means = group.groupby("instance").mean(numeric_only=True).reindex(dom_wdeg_runs["instance"]).fillna(0)
        h_time = inst_means["wall_time_sec"].values
        h_bt = inst_means["backtracks"].values

        wilcox_time = paired_wilcoxon_test(h_time, baseline_time)
        wilcox_bt = paired_wilcoxon_test(h_bt, baseline_bt)

        par2 = compute_par2(group, time_limit_sec=time_limit)
        mean_inf = float(group["inference_time_sec"].mean())

        model_seeds_bt = group.groupby("model_seed")["backtracks"].mean().values if "model_seed" in group and group["model_seed"].notnull().any() else [float(group["backtracks"].mean())]
        seed_stats = compute_multi_seed_statistics(model_seeds_bt)

        stat_records.append({
            "heuristic": h_name,
            "solve_rate_pct": round(solve_rate, 1),
            "par2_sec": round(par2, 4),
            "mean_wall_time_sec": round(float(group["wall_time_sec"].mean()), 4),
            "mean_inference_sec": round(mean_inf, 4),
            "mean_backtracks": round(float(group["backtracks"].mean()), 2),
            "bt_95ci": f"[{seed_stats['ci_lower']:.2f}, {seed_stats['ci_upper']:.2f}]",
            "wilcoxon_p_time_vs_domwdeg": round(wilcox_time["p_value"], 4),
        })

    df_stats = pd.DataFrame(stat_records).sort_values("par2_sec")

    json_path = results_dir / f"{exp_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_stats.to_dict(orient="records"), f, indent=2)

    md_table = generate_markdown_table(df_stats, title=f"Experiment E5 Time-Limited Search (Cutoff: {time_limit}s)")
    md_path = results_dir / f"{exp_name}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + md_table)

    # 5. Generate E5 plots
    try:
        plot_path_bar = results_dir / f"{exp_name}_comparison.png"
        plot_baseline_comparison(df_summary, output_path=str(plot_path_bar))
        print(f"Saved E5 comparison bar plot to: {plot_path_bar}")

        plot_path_dm = results_dir / f"{exp_name}_dolan_more.png"
        plot_dolan_more(df_runs, metric="wall_time_sec", output_path=str(plot_path_dm))
        print(f"Saved E5 Dolan-Moré performance profile to: {plot_path_dm}")
    except Exception as e:
        print(f"Plotting error: {e}")
