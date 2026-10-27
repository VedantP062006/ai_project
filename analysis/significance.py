"""Statistical significance testing and bootstrap confidence intervals."""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Any
import numpy as np
from scipy import stats


def paired_wilcoxon_test(
    scores_model: Sequence[float],
    scores_baseline: Sequence[float],
) -> Dict[str, float]:
    """
    Compute paired Wilcoxon signed-rank test between model and baseline.
    
    Returns:
        Dict with 'statistic' and 'p_value'.
    """
    diffs = np.array(scores_model) - np.array(scores_baseline)
    # If all differences are zero, p-value is 1.0
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0}

    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
        }
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0}


def bootstrap_confidence_interval(
    data: Sequence[float],
    n_bootstraps: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Calculate non-parametric bootstrap confidence interval for the mean.
    
    Args:
        data: Sequence of numerical values.
        n_bootstraps: Number of bootstrap resamples (default: 10,000).
        ci: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        (lower_bound, upper_bound)
    """
    arr = np.asarray(data)
    if len(arr) <= 1:
        val = float(arr[0]) if len(arr) == 1 else 0.0
        return val, val

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, (alpha / 2.0) * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0))
    return lower, upper


def compute_multi_seed_statistics(
    metric_values: Sequence[float],
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics across multiple training seeds:
    Mean, Median, Standard Deviation, and Bootstrap Confidence Interval.
    """
    arr = np.asarray(metric_values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci_low, ci_high = bootstrap_confidence_interval(arr, ci=ci, seed=seed)

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
    }
