"""High-performance local dashboard server and REST API for CSP Backtracking Search."""

from __future__ import annotations
import http.server
import socketserver
import json
import urllib.parse
from pathlib import Path
import time
import os
import sys

# Ensure root workspace is on pythonpath
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from solver.backtracking import BacktrackingSolver
from solver.csp import bitmask_to_values
from heuristics.variable_ordering import VARIABLE_HEURISTICS, get_top_k_mrv_candidates
from heuristics.value_ordering import VALUE_HEURISTICS
from generators.sudoku import generate_sudoku, PUBLISHED_HARD_PUZZLES, parse_sudoku_string
from generators.graph_coloring import generate_graph_coloring
from generators.cryptarithmetic import get_classic_cryptarithmetic, CLASSIC_CRYPTARITHMETIC
from analysis.metrics import aggregate_experiment_metrics

PORT = 8050
DASHBOARD_DIR = Path(__file__).resolve().parent

_LOADED_MODELS: dict[str, Any] = {}

def get_variable_heuristic(name: str):
    if name in VARIABLE_HEURISTICS:
        return VARIABLE_HEURISTICS[name]

    if name in _LOADED_MODELS:
        return _LOADED_MODELS[name]

    models_dir = BASE_DIR / "results" / "models"
    try:
        if name in ("mlp_dagger_top3", "mlp_dagger"):
            from models.mlp_model import MLPModel
            p = models_dir / "mlp_dagger_s42.pt"
            if p.exists():
                m = MLPModel.load(str(p))
                _LOADED_MODELS[name] = m.create_heuristic_fn(top_k=3)
        elif name in ("mlp_top3", "mlp"):
            from models.mlp_model import MLPModel
            p = models_dir / "mlp_s42.pt"
            if p.exists():
                m = MLPModel.load(str(p))
                _LOADED_MODELS[name] = m.create_heuristic_fn(top_k=3)
        elif name in ("lightgbm_dagger_top3", "lightgbm_dagger", "lgbm_dagger_top3"):
            from models.lightgbm_model import LightGBMModel
            p = models_dir / "lightgbm_dagger_s42.pkl"
            if p.exists():
                m = LightGBMModel.load(str(p))
                _LOADED_MODELS[name] = m.create_heuristic_fn(top_k=3)
        elif name in ("lightgbm_top3", "lightgbm", "lgbm_top3"):
            from models.lightgbm_model import LightGBMModel
            p = models_dir / "lightgbm_s42.pkl"
            if p.exists():
                m = LightGBMModel.load(str(p))
                _LOADED_MODELS[name] = m.create_heuristic_fn(top_k=3)
        elif name in ("gnn_top3", "gnn"):
            from models.gnn_model import GNNModel
            p = models_dir / "gnn_s42.pt"
            if p.exists():
                m = GNNModel()
                m.load(str(p))
                _LOADED_MODELS[name] = m.create_heuristic_fn(top_k=3)
    except Exception as e:
        print(f"Warning: Failed to load learned model {name}: {e}")

    if name in _LOADED_MODELS:
        return _LOADED_MODELS[name]

    return VARIABLE_HEURISTICS.get(name, VARIABLE_HEURISTICS["mrv_degree"])


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._send_json({"status": "running", "timestamp": time.time()})
        elif path == "/api/results":
            self._handle_get_results(parsed)
        elif path == "/api/hypotheses":
            self._handle_get_hypotheses()
        elif path == "/api/instances/list":
            self._handle_get_instance_presets()
        else:
            # Fallback to static file serving
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(post_data) if post_data else {}

        if path == "/api/solve":
            self._handle_solve(data)
        elif path == "/api/trace":
            self._handle_trace(data)
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data: Any, code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_get_results(self, parsed_url: Any = None):
        exp = "e5"
        if parsed_url and parsed_url.query:
            q = urllib.parse.parse_qs(parsed_url.query)
            exp = q.get("exp", ["e5"])[0]

        files_map = {
            "e4": ("e4_domain_transfer_results.csv", "e4_domain_transfer_summary.json"),
            "e3": ("e3_scale_shift_results.csv", "e3_scale_shift_summary.json"),
            "e5": ("e5_time_limited_results.csv", "e5_time_limited_summary.json"),
            "e2": ("e2_difficulty_shift_results.csv", "e2_difficulty_shift_summary.json"),
            "e1": ("e1_in_distribution_results.csv", "e1_in_distribution_summary.json"),
            "m1": ("m1_baselines_results.csv", "m1_baselines_summary.json"),
        }

        raw_results = []
        summary = []

        if exp in files_map:
            csv_name, json_name = files_map[exp]
            csv_path = BASE_DIR / "results" / csv_name
            json_path = BASE_DIR / "results" / json_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df = df.astype(object).where(pd.notnull(df), None)
                raw_results = df.to_dict(orient="records")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
        else:
            # Fallback to E3, E5, or E1
            for k in ["e3", "e4", "e5", "e2", "e1", "m1"]:
                c_path = BASE_DIR / "results" / files_map[k][0]
                j_path = BASE_DIR / "results" / files_map[k][1]
                if c_path.exists() and not raw_results:
                    df = pd.read_csv(c_path)
                    df = df.astype(object).where(pd.notnull(df), None)
                    raw_results = df.to_dict(orient="records")
                if j_path.exists() and not summary:
                    with open(j_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)

        self._send_json({
            "raw": raw_results,
            "summary": summary,
            "current_experiment": exp,
            "available_experiments": ["e3", "e4", "e5", "e2", "e1", "m1"],
        })

    def _handle_get_hypotheses(self):
        hypotheses = [
            {
                "id": "H1",
                "title": "Backtrack Reduction",
                "statement": "A learned variable-ordering policy reduces search tree nodes and backtracks versus MRV and dom/wdeg on held-out instances.",
                "status": "Validated on E1 (mlp_top3: 0.46 vs dom/wdeg: 0.57, mrv: 0.71)",
                "baseline_benchmark": "E1 In-Distribution: mlp_top3 achieved 0.46 mean backtracks (95% CI: [0.37, 0.54]), outperforming dom/wdeg (0.57) and MRV (0.71).",
                "target_metric": "Mean Backtracks & Search Nodes on Held-out Sets",
            },
            {
                "id": "H2",
                "title": "Survival Over Inference Cost",
                "statement": "The reduction in backtracks survives model inference cost and improves solve rate within a fixed time limit.",
                "status": "Strongly Validated on E5 (100% Solve Rate, 0.0087s Inference Overhead)",
                "baseline_benchmark": "E5 Time-Limited: mlp_dagger_top3 retained 100% solve rate with 10.35 mean backtracks vs 52.58 for dom/wdeg. Classical degree & random timed out (94.7% solve rate, PAR2 > 0.70s).",
                "target_metric": "PAR2 Score & Wall-clock Solve Rate under 5s timeout",
            },
            {
                "id": "H3",
                "title": "Out-of-Distribution Generalisation",
                "statement": "The policy generalises to harder or larger instances than seen in training (e.g. 9x9 -> 16x16 Sudoku, V=30 -> V=60 graphs).",
                "status": "Validated on E2 (Difficulty) & E3 (Scale Shift to 16x16 Hexadoku)",
                "baseline_benchmark": "E2 & E3: In E2, DAgger cut backtracks to 55.92 on hard puzzles. In E3, models scaled to 256-variable Hexadoku (lgbm_top3: 2.6 bt vs MRV: 10.92 bt, 100% solve rate).",
                "target_metric": "Generalization Backtrack Gap Closed (%)",
            },
            {
                "id": "H4",
                "title": "Top-k MRV Tie-Breaking Advantage",
                "statement": "Using the model only to break ties among the top-k MRV candidates beats using it at every node.",
                "status": "Strongly Validated on E1 (mlp_top3: 0.46 vs mlp_full: 31.83)",
                "baseline_benchmark": "E1 Ablation: Evaluating MLP on all variables expanded 31.83 backtracks; restricting to top-3 MRV tie-breaker achieved 0.46 backtracks (69x reduction).",
                "target_metric": "Inference time overhead reduction vs. Backtrack quality",
            },
        ]
        self._send_json(hypotheses)

    def _handle_get_instance_presets(self):
        presets = {
            "sudoku": [
                {"id": "sudoku_easy_1", "name": "Sudoku Easy (38 clues)", "difficulty": "easy", "seed": 100},
                {"id": "sudoku_med_1", "name": "Sudoku Medium (32 clues)", "difficulty": "medium", "seed": 200},
                {"id": "sudoku_hard_1", "name": "Sudoku Hard (26 clues)", "difficulty": "hard", "seed": 300},
                {"id": "ai_escargot", "name": "Published: AI Escargot (Arto Inkala)", "difficulty": "hard", "puzzle_key": "ai_escargot"},
                {"id": "golden_nugget", "name": "Published: Golden Nugget", "difficulty": "hard", "puzzle_key": "golden_nugget"},
            ],
            "graph_coloring": [
                {"id": "graph_20_3", "name": "Graph Coloring 20 Vertices, 3 Colors", "num_vertices": 20, "num_colors": 3, "avg_degree": 4.2},
                {"id": "graph_30_3", "name": "Graph Coloring 30 Vertices, 3 Colors (Critical)", "num_vertices": 30, "num_colors": 3, "avg_degree": 4.5},
                {"id": "graph_40_3", "name": "Graph Coloring 40 Vertices, 3 Colors", "num_vertices": 40, "num_colors": 3, "avg_degree": 4.6},
            ],
            "cryptarithmetic": [
                {"id": "send_more_money", "name": "SEND + MORE = MONEY"},
                {"id": "cross_roads_danger", "name": "CROSS + ROADS = DANGER"},
                {"id": "donald_gerald_robert", "name": "DONALD + GERALD = ROBERT"},
            ],
        }
        self._send_json(presets)

    def _build_instance(self, data: Dict[str, Any]):
        family = data.get("family", "sudoku")
        seed = int(data.get("seed", 42))

        if family == "sudoku":
            puzzle_key = data.get("puzzle_key")
            if puzzle_key and puzzle_key in PUBLISHED_HARD_PUZZLES:
                return parse_sudoku_string(PUBLISHED_HARD_PUZZLES[puzzle_key], name=puzzle_key, difficulty="hard")
            diff = data.get("difficulty", "medium")
            return generate_sudoku(seed=seed, difficulty=diff)

        elif family == "graph_coloring":
            num_v = int(data.get("num_vertices", 25))
            num_k = int(data.get("num_colors", 3))
            avg_deg = float(data.get("avg_degree", 4.4))
            return generate_graph_coloring(num_vertices=num_v, num_colors=num_k, avg_degree=avg_deg, planted=True, seed=seed)

        elif family == "cryptarithmetic":
            puzzle_key = data.get("puzzle_key", "send_more_money")
            return get_classic_cryptarithmetic(puzzle_key)

        raise ValueError(f"Unknown problem family {family}")

    def _handle_solve(self, data: Dict[str, Any]):
        try:
            inst = self._build_instance(data)
            h_name = data.get("heuristic", "mlp_dagger_top3")
            var_h = get_variable_heuristic(h_name)
            val_h = VALUE_HEURISTICS.get(data.get("value_heuristic", "natural"), VALUE_HEURISTICS["natural"])
            prop_mode = data.get("propagation_mode", "mac")
            time_limit = float(data.get("time_limit_sec", 10.0))

            solver = BacktrackingSolver(
                variable_heuristic=var_h,
                value_heuristic=val_h,
                propagation_mode=prop_mode,
                time_limit_sec=time_limit,
                verify=True,
            )

            sol, stats = solver.solve(inst.csp, seed=int(data.get("seed", 42)))

            # Format domains / initial board for display
            initial_board = []
            for v in range(inst.csp.num_variables):
                initial_board.append({
                    "var": v,
                    "name": inst.csp.variable_names[v],
                    "domain_size": inst.csp.domain_size(v),
                    "values": inst.csp.domain_values(v),
                    "assigned_val": inst.csp.assigned_value(v),
                })

            response = {
                "instance_name": inst.name,
                "family": inst.family,
                "difficulty": inst.difficulty,
                "stats": stats.to_dict(),
                "initial_board": initial_board,
                "solution": sol,
                "num_variables": inst.csp.num_variables,
                "metadata": inst.metadata,
            }
            self._send_json(response)
        except Exception as e:
            self._send_json({"error": str(e)}, code=400)

    def _handle_trace(self, data: Dict[str, Any]):
        """Execute search while recording visual step events for playback animation."""
        try:
            inst = self._build_instance(data)
            h_name = data.get("heuristic", "mlp_dagger_top3")
            var_h = get_variable_heuristic(h_name)
            val_h = VALUE_HEURISTICS.get(data.get("value_heuristic", "natural"), VALUE_HEURISTICS["natural"])
            prop_mode = data.get("propagation_mode", "mac")
            max_events = int(data.get("max_events", 200))

            events: List[Dict[str, Any]] = []
            csp = inst.csp

            # Custom trace-recording search
            trail = []
            nodes = 0
            backtracks = 0

            # Initial propagation
            ok = solver_ac3 = True
            from solver.propagation import ac3_propagate, forward_checking
            ac3_propagate(csp, trail=trail)

            events.append({
                "type": "INITIAL_PROPAGATION",
                "message": "Established initial Arc Consistency (MAC / AC-3).",
                "domains": [csp.domain_values(v) for v in range(csp.num_variables)],
            })

            def record_search(depth: int) -> bool:
                nonlocal nodes, backtracks
                nodes += 1
                if len(events) >= max_events:
                    return False

                if csp.is_complete():
                    events.append({
                        "type": "SOLVED",
                        "depth": depth,
                        "nodes": nodes,
                        "backtracks": backtracks,
                        "message": "Complete consistent assignment found!",
                        "assignment": csp.current_assignment(),
                    })
                    return True

                var = var_h(csp)
                if var == -1 or csp.is_wiped_out(var):
                    backtracks += 1
                    return False

                candidates = get_top_k_mrv_candidates(csp, k=3)
                events.append({
                    "type": "SELECT_VAR",
                    "var": var,
                    "var_name": csp.variable_names[var],
                    "domain_size": csp.domain_size(var),
                    "domain_values": csp.domain_values(var),
                    "candidates": candidates,
                    "depth": depth,
                    "nodes": nodes,
                    "backtracks": backtracks,
                })

                vals = val_h(csp, var)
                for val in vals:
                    trail_mark = len(trail)
                    pruned = csp.domains[var] ^ (1 << val)
                    csp.domains[var] = (1 << val)
                    trail.append((var, pruned))

                    events.append({
                        "type": "ASSIGN",
                        "var": var,
                        "var_name": csp.variable_names[var],
                        "val": val,
                        "depth": depth,
                    })

                    if prop_mode == "mac":
                        arcs = [(k, var) for k in csp.neighbors[var]]
                        prop_ok = ac3_propagate(csp, arcs=arcs, trail=trail, track_weights=True)
                    else:
                        prop_ok = forward_checking(csp, assigned_var=var, trail=trail, track_weights=True)

                    events.append({
                        "type": "PROPAGATE",
                        "var": var,
                        "val": val,
                        "success": prop_ok,
                        "depth": depth,
                        "active_domains": [csp.domain_values(v) for v in range(min(csp.num_variables, 81))],
                    })

                    if prop_ok:
                        if record_search(depth + 1):
                            return True

                    backtracks += 1
                    while len(trail) > trail_mark:
                        v_u, m_u = trail.pop()
                        csp.domains[v_u] |= m_u

                    events.append({
                        "type": "BACKTRACK",
                        "var": var,
                        "val": val,
                        "depth": depth,
                        "backtracks": backtracks,
                    })

                return False

            record_search(0)

            # Extract initial layout
            graph_data = None
            if inst.family == "graph_coloring":
                graph_data = {
                    "num_vertices": inst.metadata.get("num_vertices", 25),
                    "edges": list(inst.csp.neighbors),
                }

            self._send_json({
                "instance_name": inst.name,
                "family": inst.family,
                "difficulty": inst.difficulty,
                "num_variables": csp.num_variables,
                "events": events,
                "graph_data": graph_data,
                "metadata": inst.metadata,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, code=400)


def run_server():
    os.chdir(str(DASHBOARD_DIR))
    server_address = ("", PORT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, DashboardHandler) as httpd:
        print(f"CSP Research Dashboard serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
