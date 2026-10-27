# Learning Variable and Value Ordering for CSP Backtracking Search

## Research Question
Can a learned model choose variable and value orderings in backtracking search with constraint propagation (MAC / AC-3) that beat classical heuristics (MRV, degree, dom/wdeg) in backtracks, runtime, and solve rate under a time limit?

---

## Hypotheses
Before running experiments, the following hypotheses are established to guide evaluation:

* **H1 (Backtrack Reduction):** A learned variable-ordering policy reduces search tree nodes and backtracks versus classical heuristics (MRV, degree, and dom/wdeg) on held-out test instances.
* **H2 (Survival of Inference Cost):** The reduction in backtracks is sufficient to overcome model evaluation overhead, improving solve rate and PAR2 score within a fixed per-instance time limit (default: 10 s).
* **H3 (Out-of-Distribution Generalisation):** A policy trained on small or easy instances generalises effectively to harder difficulty levels (e.g., Easy/Medium $\to$ Hard Sudoku) and larger problem structures (e.g., $9 \times 9 \to 16 \times 16$ Sudoku, $V=30 \to V=60$ Graph Colouring).
* **H4 (Hybrid Tie-Breaking Advantage):** Using the learned model only to break ties among the top-$k$ candidates nominated by MRV ($k \in \{2, 3, 5\}$) dominates evaluating the model across all unassigned variables at every search node.

---

## Tech Stack & Architecture
- **Language:** Python 3.11+
- **Numerical & ML:** NumPy, SciPy, Pandas, PyTorch, LightGBM, NetworkX, Matplotlib
- **Sanity Checking:** Google OR-Tools (CP-SAT) exclusively as an external ground-truth validation tool; never used in the learned solver.
- **Hardware Profile:** Fully runnable on laptop CPU; GPU optional.

---

## Repository Layout
```
solver/        - CSP representation (bitmask domains), AC-3 propagation, backtracking search, undo trail, instrumentation, verifier
heuristics/    - Variable orderings (static, random, MRV, degree, MRV+degree, dom/wdeg, top-k filter); Value orderings (random, natural, LCV)
generators/    - Problem generators and loaders: Sudoku (9x9, 16x16, hard benchmarks), Graph Colouring (phase transition), Cryptarithmetic
labels/        - Oracle generation: planted solutions, bounded lookahead (strong branching), MRV imitation
features/      - High-speed dynamic per-variable features (domain size, degree, dom/wdeg, failure counts, graph metrics)
models/        - LightGBM ranker, PyTorch MLP, Bipartite Graph Neural Network (GNN)
training/      - Supervised learning pipeline and DAgger iterative imitation loop
experiments/   - Unified experiment execution engine (python -m experiments.run --config configs/<name>.yaml)
analysis/      - Statistical testing (paired Wilcoxon), Dolan-Moré performance profiles, bootstrap CIs, LaTeX/Markdown tables
configs/       - Reproducible YAML experiment specifications
results/       - Saved CSV trajectories, JSON summaries, and generated publication figures
tests/         - Comprehensive pytest test suite
```

---

## Evaluation Protocol
1. **Instance Isolation:** Train, validation, and test splits are strictly partitioned by problem instance, never by search node.
2. **Leakage Elimination:** Features are strictly computed from the unassigned search state without access to ground truth solutions.
3. **Multi-Seed Stability:** Minimum 5 independent seeds per learned policy; metrics reported as mean, median, and 95% bootstrap confidence intervals.
4. **Metrics:**
   - **Solve Rate (%):** Percentage of test instances solved within time limit.
   - **Nodes & Backtracks:** Direct measure of search tree efficiency.
   - **Wall-Clock Time:** Real runtime including inference overhead.
   - **PAR2 (Penalised Average Runtime 2):** Solved instances scored at runtime; timeouts scored at $2 \times \text{time\_limit}$.
   - **Oracle Gap Closed (%):** Fraction of backtrack gap between MRV and lookahead oracle closed by the model.
5. **Statistical Significance:** Paired Wilcoxon signed-rank test against the strongest baseline (`dom/wdeg`).
