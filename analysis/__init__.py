"""Analysis, metrics, plotting, and statistical reporting package."""

from analysis.metrics import (
    compute_par2,
    compute_oracle_gap_closed,
    aggregate_experiment_metrics,
)
from analysis.plots import (
    plot_dolan_more,
    plot_baseline_comparison,
)
from analysis.tables import (
    generate_markdown_table,
    generate_latex_table,
)

__all__ = [
    "compute_par2",
    "compute_oracle_gap_closed",
    "aggregate_experiment_metrics",
    "plot_dolan_more",
    "plot_baseline_comparison",
    "generate_markdown_table",
    "generate_latex_table",
]
