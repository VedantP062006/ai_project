"""Table generation utilities in Markdown and LaTeX format without required external dependencies."""

from __future__ import annotations
import pandas as pd


def generate_markdown_table(df: pd.DataFrame, title: str = "Experimental Results") -> str:
    """Format DataFrame as a clean Markdown table with title."""
    try:
        content = df.to_markdown(index=False)
    except (ImportError, Exception):
        # Fallback pure-python markdown formatter
        headers = [str(col) for col in df.columns]
        rows = [[str(val) for val in row] for row in df.values]
        
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))
        
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        sep_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
        row_lines = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, col_widths)) + " |" for row in rows]
        content = "\n".join([header_line, sep_line] + row_lines)

    md = f"### {title}\n\n{content}\n"
    return md


def generate_latex_table(df: pd.DataFrame, caption: str = "Experimental Results", label: str = "tab:results") -> str:
    """Format DataFrame as a publication-ready LaTeX table."""
    try:
        latex = df.to_latex(index=False, caption=caption, label=label, float_format="%.4f")
    except Exception:
        latex = df.to_string(index=False)
    return latex
