"""Run trained variable ordering models and benchmark them on CSP instances."""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.backtracking import BacktrackingSolver
from solver.verifier import verify_solution
from heuristics.variable_ordering import VARIABLE_HEURISTICS
from heuristics.value_ordering import VALUE_HEURISTICS
from generators.sudoku import generate_sudoku, parse_sudoku_string, PUBLISHED_HARD_PUZZLES
from generators.graph_coloring import generate_graph_coloring
from generators.cryptarithmetic import get_classic_cryptarithmetic
from models.mlp_model import MLPModel
from models.lightgbm_model import LightGBMModel
from models.gnn_model import GNNModel


def load_model(name: str):
    models_dir = ROOT_DIR / "results" / "models"
    if name == "mlp_dagger":
        return MLPModel.load(str(models_dir / "mlp_dagger_s42.pt"))
    elif name == "mlp":
        return MLPModel.load(str(models_dir / "mlp_s42.pt"))
    elif name == "lightgbm_dagger":
        return LightGBMModel.load(str(models_dir / "lightgbm_dagger_s42.pkl"))
    elif name == "lightgbm":
        return LightGBMModel.load(str(models_dir / "lightgbm_s42.pkl"))
    elif name == "gnn":
        m = GNNModel()
        m.load(str(models_dir / "gnn_s42.pt"))
        return m
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_test_instance(inst_type: str):
    if inst_type == "sudoku_hard":
        return generate_sudoku(seed=300, difficulty="hard", name="Sudoku Hard (Seed 300)")
    elif inst_type == "ai_escargot":
        raw = PUBLISHED_HARD_PUZZLES["ai_escargot"]
        return parse_sudoku_string(raw, name="AI Escargot (Extreme Benchmark)", difficulty="extreme")
    elif inst_type == "graph_coloring":
        return generate_graph_coloring(num_vertices=30, num_colors=3, avg_degree=4.5, planted=True, seed=200, name="Graph Coloring (Phase Transition V=30)")
    elif inst_type == "cryptarithmetic":
        return get_classic_cryptarithmetic("send_more_money")
    else:
        return generate_sudoku(seed=100, difficulty="medium", name="Sudoku Medium (Seed 100)")


def run_benchmark(model_keys=None, instances=None):
    if model_keys is None:
        model_keys = ["mlp_dagger", "lightgbm_dagger", "gnn"]
    if instances is None:
        instances = ["sudoku_hard", "ai_escargot", "graph_coloring", "cryptarithmetic"]

    print("=" * 80)
    print("  CSP LEARNED VARIABLE ORDERING & SEARCH BENCHMARK")
    print("=" * 80)

    # Preload models
    models = {}
    for k in model_keys:
        try:
            print(f"Loading model: {k}...", end=" ", flush=True)
            models[k] = load_model(k)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    policies = []
    # Learned models (top-3 tie-breaking)
    for k, m in models.items():
        policies.append((f"{k}_top3", m.create_heuristic_fn(top_k=3), True))
    # Baselines
    policies.append(("dom_wdeg (classical)", VARIABLE_HEURISTICS["dom_wdeg"], False))
    policies.append(("mrv_degree (classical)", VARIABLE_HEURISTICS["mrv_degree"], False))
    policies.append(("random (uninformed)", VARIABLE_HEURISTICS["random"], False))

    for inst_key in instances:
        inst = get_test_instance(inst_key)
        print(f"\n--- Instance: {inst.name} ({inst.family.upper()}, vars={inst.csp.num_variables}) ---")
        header = f"{'Policy / Model':<28} | {'Status':<7} | {'Nodes':>6} | {'Backtracks':>10} | {'Time (s)':>10} | {'Verified':>8}"
        print(header)
        print("-" * len(header))

        for name, h_fn, is_learned in policies:
            csp_clone = inst.csp.clone()
            solver = BacktrackingSolver(
                variable_heuristic=h_fn,
                value_heuristic=VALUE_HEURISTICS["natural"],
                propagation_mode="mac",
                time_limit_sec=10.0,
                verify=True,
            )

            sol, stats = solver.solve(csp_clone)
            verified = "YES" if sol and verify_solution(inst.csp, sol) else ("NO" if sol else "TIMEOUT")
            print(f"{name:<28} | {stats.status:<7} | {stats.nodes:>6} | {stats.backtracks:>10} | {stats.wall_time_sec:>10.4f} | {verified:>8}")

    print("\n" + "=" * 80)
    print("Benchmark complete! All model solutions verified.")
    print("Interactive visualizer is available at: http://localhost:8050")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained CSP variable ordering models.")
    parser.add_argument("--model", type=str, default="all", choices=["all", "mlp_dagger", "mlp", "lightgbm_dagger", "lightgbm", "gnn"], help="Model to evaluate")
    parser.add_argument("--instance", type=str, default="all", choices=["all", "sudoku_hard", "ai_escargot", "graph_coloring", "cryptarithmetic"], help="Instance to solve")
    args = parser.parse_args()

    m_keys = ["mlp_dagger", "lightgbm_dagger", "gnn", "mlp", "lightgbm"] if args.model == "all" else [args.model]
    insts = ["sudoku_hard", "ai_escargot", "graph_coloring"] if args.instance == "all" else [args.instance]

    run_benchmark(model_keys=m_keys, instances=insts)
