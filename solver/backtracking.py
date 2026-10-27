"""Backtracking search engine with Maintain Arc Consistency (MAC) and undo trail."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any
import time
import random

from solver.csp import CSP
from solver.propagation import ac3_propagate, forward_checking
from solver.instrumentation import SearchStats, InstrumentationTimer
from solver.verifier import verify_solution


VariableOrderingFn = Callable[[CSP, Any, InstrumentationTimer], int]
ValueOrderingFn = Callable[[CSP, int, Any, InstrumentationTimer], List[int]]


class BacktrackingSolver:
    """
    Instrumented backtracking search with Maintain Arc Consistency (MAC) or Forward Checking (FC).
    
    Features:
    - Zero-allocation domain backtracking via an undo trail of bitmasks.
    - Pluggable variable and value ordering heuristics.
    - Strict per-instance timeout and deterministic PRNG seeding.
    - Separate tracking of search and model-inference times.
    - Loud solution verification on completion.
    """

    def __init__(
        self,
        variable_heuristic: Optional[VariableOrderingFn] = None,
        value_heuristic: Optional[ValueOrderingFn] = None,
        propagation_mode: str = "mac",
        time_limit_sec: float = 10.0,
        verify: bool = True,
    ) -> None:
        self.variable_heuristic = variable_heuristic
        self.value_heuristic = value_heuristic
        self.propagation_mode = propagation_mode.lower()
        if self.propagation_mode not in ("mac", "fc"):
            raise ValueError(f"Unknown propagation_mode '{propagation_mode}'. Choose 'mac' or 'fc'.")
        self.time_limit_sec = time_limit_sec
        self.verify = verify

    def solve(
        self,
        csp: CSP,
        seed: int = 42,
        model: Optional[Any] = None,
    ) -> Tuple[Optional[Dict[int, int]], SearchStats]:
        """
        Solve the given CSP instance.
        
        Args:
            csp: CSP instance to solve.
            seed: Random seed for deterministic tie-breaking and random heuristics.
            model: Optional ML model / policy passed to heuristics.
            
        Returns:
            Tuple of (solution_dict, search_stats).
        """
        random.seed(seed)
        timer = InstrumentationTimer()
        timer.start_wall()

        stats = SearchStats(
            seed=seed,
            time_limit_sec=self.time_limit_sec,
            instance_name=csp.name,
        )

        trail: List[Tuple[int, int]] = []
        deadline = timer.wall_start + self.time_limit_sec

        # 1. Initial propagation to establish arc consistency before search
        stats.propagation_calls += 1
        initial_ok = ac3_propagate(csp, trail=trail, track_weights=True)
        if not initial_ok:
            stats.status = "INFEASIBLE"
            stats.wall_time_sec = timer.stop_wall()
            stats.inference_time_sec = timer.total_inference_time
            # Rollback trail
            while trail:
                var, mask = trail.pop()
                csp.domains[var] |= mask
            return None, stats

        # 2. Check if already complete after initial propagation
        if csp.is_complete():
            sol = csp.current_assignment()
            if self.verify:
                verify_solution(csp, sol, raise_on_failure=True)
            stats.status = "SOLVED"
            stats.wall_time_sec = timer.stop_wall()
            stats.inference_time_sec = timer.total_inference_time
            return sol, stats

        # 3. Recursive backtracking search loop
        def backtrack(depth: int) -> Optional[Dict[int, int]]:
            stats.nodes += 1
            if depth > stats.max_depth:
                stats.max_depth = depth

            # Check timeout every 10 nodes for high throughput
            if (stats.nodes % 10 == 0) and (time.perf_counter() >= deadline):
                stats.status = "TIMEOUT"
                return None

            if csp.is_complete():
                return csp.current_assignment()

            # Select unassigned variable
            if self.variable_heuristic is not None:
                var = self.variable_heuristic(csp, model, timer)
            else:
                # Default MRV fallback
                var = -1
                min_size = 999999
                for v in range(csp.num_variables):
                    sz = csp.domain_size(v)
                    if 1 < sz < min_size:
                        min_size = sz
                        var = v

            if var == -1 or csp.is_wiped_out(var):
                stats.backtracks += 1
                return None

            # Order values for selected variable
            if self.value_heuristic is not None:
                values = self.value_heuristic(csp, var, model, timer)
            else:
                values = csp.domain_values(var)

            for val in values:
                trail_mark = len(trail)
                current_mask = csp.domains[var]

                # Assign var <- val
                pruned = current_mask ^ (1 << val)
                csp.domains[var] = (1 << val)
                trail.append((var, pruned))

                # Propagate
                stats.propagation_calls += 1
                if self.propagation_mode == "mac":
                    arcs = [(k, var) for k in csp.neighbors[var]]
                    prop_ok = ac3_propagate(csp, arcs=arcs, trail=trail, track_weights=True)
                else:  # Forward checking
                    prop_ok = forward_checking(csp, assigned_var=var, trail=trail, track_weights=True)

                if prop_ok:
                    result = backtrack(depth + 1)
                    if result is not None:
                        return result
                    if stats.status == "TIMEOUT":
                        # Unwind trail on timeout
                        while len(trail) > trail_mark:
                            v_undo, m_undo = trail.pop()
                            csp.domains[v_undo] |= m_undo
                        return None

                # Propagation failed or subtree had no solution: Backtrack
                stats.backtracks += 1
                while len(trail) > trail_mark:
                    v_undo, m_undo = trail.pop()
                    csp.domains[v_undo] |= m_undo

            return None

        solution = backtrack(depth=0)

        # Unwind any initial propagation entries to leave CSP clean
        while trail:
            var, mask = trail.pop()
            csp.domains[var] |= mask

        stats.wall_time_sec = timer.stop_wall()
        stats.inference_time_sec = timer.total_inference_time

        if stats.status == "TIMEOUT":
            return None, stats

        if solution is not None:
            stats.status = "SOLVED"
            if self.verify:
                verify_solution(csp, solution, raise_on_failure=True)
            return solution, stats

        stats.status = "INFEASIBLE"
        return None, stats
