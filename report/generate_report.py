#!/usr/bin/env python3
"""
report/generate_report.py

Reads all pipeline output files and auto-populates the LaTeX results
tables in main.tex, producing a complete results_auto.tex file that
is \input{}'d by main.tex.

Usage:
    python report/generate_report.py

Outputs:
    report/results_auto.tex   — LaTeX \newcommand definitions for every metric
    report/tables/            — Standalone .tex table files
    report/main_filled.tex    — Fully populated version of main.tex (for inspection)

Scientific Rigor:
    - All values read directly from pipeline output (no hard-coding)
    - Significant figures: 4 decimal places for all metrics
    - Coverage and shortfall rates formatted as percentages with 2 dp
    - SHAP top-10 table populated from xgb_shap_importance.csv
    - CV fold table populated from xgb_cv_results.parquet
    - Model comparison table from model_comparison.csv
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
REPORT_DIR    = Path("report")
TABLES_DIR    = REPORT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(value, decimals: int = 4, pct: bool = False) -> str:
    """Format a float for LaTeX. Returns '--' if None or NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return r"\textemdash"
    if pct:
        return f"{value * 100:.2f}\\%"
    return f"{value:.{decimals}f}"


def bold_best(values: list, minimize: bool = True) -> list:
    """Return list of strings with the best value bolded."""
    numeric = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    if not numeric:
        return [str(v) for v in values]
    best = min(numeric) if minimize else max(numeric)
    return [
        f"\\textbf{{{fmt(v)}}}" if (isinstance(v, (int, float)) and not np.isnan(v) and v == best)
        else fmt(v)
        for v in values
    ]


# ── Load pipeline outputs ─────────────────────────────────────────────────────

def load_model_comparison() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "model_comparison.csv"
    if not p.exists():
        print(f"  [WARN] Not found: {p}")
        return None
    return pd.read_csv(p)


def load_xgb_cv() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "xgb_cv_results.parquet"
    if not p.exists():
        print(f"  [WARN] Not found: {p}")
        return None
    return pd.read_parquet(p)


def load_xgb_summary() -> dict | None:
    p = PROCESSED_DIR / "xgb_summary.json"
    if not p.exists():
        print(f"  [WARN] Not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_shap_importance() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "xgb_shap_importance.csv"
    if not p.exists():
        print(f"  [WARN] Not found: {p}")
        return None
    return pd.read_csv(p)


def load_shortfall() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "shortfall_risk.parquet"
    if not p.exists():
        print(f"  [WARN] Not found: {p}")
        return None
    return pd.read_parquet(p)


def load_anomalies() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "anomalies.parquet"
    if not p.exists():
        print(f"  [WARN] Not found: {p}")
        return None
    return pd.read_parquet(p)


# ── Table generators ──────────────────────────────────────────────────────────

def generate_model_comparison_table(comparison: pd.DataFrame) -> str:
    """Generate the full model comparison LaTeX table."""
    metric_cols = ["mape", "smape", "mae", "rmse", "wrmsse", "coverage_95", "bias"]
    avail_cols  = [c for c in metric_cols if c in comparison.columns]

    header_map = {
        "mape":        r"\textbf{MAPE}$\downarrow$",
        "smape":       r"\textbf{sMAPE}$\downarrow$",
        "mae":         r"\textbf{MAE}$\downarrow$",
        "rmse":        r"\textbf{RMSE}$\downarrow$",
        "wrmsse":      r"\textbf{WRMSSE}$\downarrow$",
        "coverage_95": r"\textbf{Coverage}$\uparrow$",
        "bias":        r"\textbf{Bias}",
    }
    minimize_map = {
        "mape": True, "smape": True, "mae": True,
        "rmse": True, "wrmsse": True,
        "coverage_95": False,   # higher is better
        "bias": False,          # closer to 0 is better — handled manually
    }

    col_header = " & ".join(
        [r"\textbf{Model}"] + [header_map.get(c, c) for c in avail_cols]
    ) + r" \\"

    rows_tex = []
    for _, row in comparison.iterrows():
        cells = [row.get("Model", "Unknown")]
        for col in avail_cols:
            v = row.get(col)
            if isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    pass
            cells.append(v if isinstance(v, (int, float)) else None)

        # Bold best per column
        col_vals = [
            comparison[col].apply(lambda x: float(x) if isinstance(x, str) else x).tolist()
            for col in avail_cols
        ]
        row_tex = row.get("Model", "?")
        for idx, col in enumerate(avail_cols):
            v = cells[idx + 1]
            col_all = [
                float(c) if isinstance(c, str) else c
                for c in comparison[col].tolist()
            ]
            minimize = minimize_map.get(col, True)
            if isinstance(v, (int, float)) and not np.isnan(float(v)):
                is_best = (float(v) == min(col_all) if minimize else float(v) == max(col_all))
                formatted = fmt(float(v), pct=(col == "coverage_95"))
                row_tex += " & " + (f"\\textbf{{{formatted}}}" if is_best else formatted)
            else:
                row_tex += r" & \textemdash"
        row_tex += r" \\"
        rows_tex.append(row_tex)

    n_cols = 1 + len(avail_cols)
    col_spec = "@{}" + "l" + "c" * len(avail_cols) + "@{}"

    table = r"""\begin{table}[H]
\centering
\caption{Model comparison on 28-day hold-out (walk-forward fold 5, representative 30-series subset). Bold = best per column.}
\label{tab:model_comparison}
\begin{tabular}{""" + col_spec + r"""}
\toprule
""" + col_header + r"""
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\multicolumn{""" + str(n_cols) + r"""}{l}{\footnotesize MAPE/sMAPE/MAE/RMSE: lower is better. Coverage: higher is better (target 0.95).}\\
\end{tabular}
\end{table}
"""
    return table


def generate_cv_table(cv: pd.DataFrame) -> str:
    """Generate the XGBoost CV fold results table."""
    rows = []
    for _, row in cv.iterrows():
        rows.append(
            f"{int(row['fold'])} & {row['train_end']} & "
            f"{fmt(row['mape'])} & {fmt(row['rmse'])} & "
            f"{int(row.get('n_train', 0)):,} \\\\"
        )
    # Summary row
    rows.append(r"\midrule")
    rows.append(
        f"\\textbf{{Mean}} & --- & "
        f"\\textbf{{{fmt(cv['mape'].mean())}}} & "
        f"\\textbf{{{fmt(cv['rmse'].mean())}}} & --- \\\\"
    )
    rows.append(
        f"\\textbf{{Std}}  & --- & "
        f"\\textbf{{{fmt(cv['mape'].std())}}} & "
        f"\\textbf{{{fmt(cv['rmse'].std())}}}  & --- \\\\"
    )

    table = r"""\begin{table}[H]
\centering
\caption{XGBoost walk-forward CV results (5 folds, 28-day test horizon each)}
\label{tab:xgb_cv}
\begin{tabular}{@{}cllcr@{}}
\toprule
\textbf{Fold} & \textbf{Train End} & \textbf{MAPE}$\downarrow$ & \textbf{RMSE}$\downarrow$ & \textbf{N Train} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\multicolumn{5}{l}{\footnotesize From \texttt{data/processed/xgb\_cv\_results.parquet}.}
\end{tabular}
\end{table}
"""
    return table


def generate_shap_table(shap: pd.DataFrame) -> str:
    """Generate top-10 SHAP feature importance table."""
    top10 = shap.head(10).reset_index(drop=True)

    rows = []
    for i, row in top10.iterrows():
        feat = row["feature"].replace("_", r"\_")
        rows.append(
            f"{i + 1} & \\texttt{{{feat}}} & {fmt(row['mean_abs_shap'])} \\\\"
        )

    table = r"""\begin{table}[H]
\centering
\caption{Top-10 XGBoost features by mean $|\text{SHAP}|$ value (computed on 2{,}000-row test sample)}
\label{tab:shap}
\begin{tabular}{@{}clc@{}}
\toprule
\textbf{Rank} & \textbf{Feature} & \textbf{Mean $|\text{SHAP}|$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\multicolumn{3}{l}{\footnotesize From \texttt{data/processed/xgb\_shap\_importance.csv}.}
\end{tabular}
\end{table}
"""
    return table


def generate_coverage_table(shortfall: pd.DataFrame, xgb_summary: dict | None) -> str:
    """Generate SRE coverage calibration table."""
    if "shortfall_breach" in shortfall.columns:
        actual_rate = shortfall["shortfall_breach"].mean()
        cal_error   = abs(actual_rate - 0.05)
    else:
        actual_rate = float("nan")
        cal_error   = float("nan")

    cov_95 = fmt(xgb_summary.get("coverage_95"), pct=True) if xgb_summary else r"\textemdash"

    table = r"""\begin{table}[H]
\centering
\caption{SRE coverage backtest results ($\alpha=0.05$, window $w=90$ days)}
\label{tab:coverage}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Model CI} & \textbf{Target $\alpha$} & \textbf{Actual Breach Rate} & \textbf{Calibration Error} \\
\midrule
XGBoost 95\% CI  & $5.00\%$ & """ + fmt(actual_rate, pct=True) + r""" & """ + fmt(cal_error, pct=True) + r""" \\
SRE ($w=90$)     & $5.00\%$ & """ + fmt(actual_rate, pct=True) + r""" & """ + fmt(cal_error, pct=True) + r""" \\
\bottomrule
\multicolumn{4}{l}{\footnotesize From \texttt{src/risk/shortfall.py:coverage\_backtest()}.}
\end{tabular}
\end{table}
"""
    return table


def generate_commands(comparison, cv, xgb_summary, shortfall, anomalies) -> str:
    """
    Generate \\newcommand definitions for all key metrics so they can
    be cited anywhere in the document with e.g. \\XgbMape.
    """
    lines = ["% AUTO-GENERATED — do not edit manually",
             "% Run: python report/generate_report.py",
             ""]

    def defcmd(name: str, value) -> str:
        return f"\\newcommand{{\\{name}}}{{{fmt(value)}}}"

    def defpct(name: str, value) -> str:
        if value is None or np.isnan(float(value)):
            return f"\\newcommand{{\\{name}}}{{--}}"
        return f"\\newcommand{{\\{name}}}{{{float(value)*100:.2f}\\%}}"

    if xgb_summary:
        lines += [
            defcmd("XgbMape",        xgb_summary.get("mape")),
            defcmd("XgbRmse",        xgb_summary.get("rmse")),
            defpct("XgbCoverage",    xgb_summary.get("coverage_95")),
            defcmd("XgbWidth",       xgb_summary.get("interval_width")),
            defcmd("XgbCvMapeMean",  xgb_summary.get("cv_mape_mean")),
            defcmd("XgbCvMapeStd",   xgb_summary.get("cv_mape_std")),
            defcmd("XgbCvRmseMean",  xgb_summary.get("cv_rmse_mean")),
            defcmd("XgbCvRmseStd",   xgb_summary.get("cv_rmse_std")),
        ]

    if comparison is not None:
        for _, row in comparison.iterrows():
            model_tag = str(row.get("Model", "Unknown")).replace("-", "")
            for metric in ["mape", "rmse", "mae", "smape", "coverage_95"]:
                v = row.get(metric)
                if v is not None:
                    cmd_name = model_tag + metric.replace("_", "").capitalize()
                    if metric == "coverage_95":
                        lines.append(defpct(cmd_name, v))
                    else:
                        lines.append(defcmd(cmd_name, v))

    if shortfall is not None and "shortfall_breach" in shortfall.columns:
        breach = shortfall["shortfall_breach"].mean()
        lines.append(defpct("ShortfallBreachRate", breach))
        lines.append(defcmd("ShortfallCalibError", abs(breach - 0.05)))

    if anomalies is not None:
        lines.append(f"\\newcommand{{\\TotalAnomalyFlags}}{{{len(anomalies):,}}}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Report Generator — Retail Demand Forecasting Platform")
    print("=" * 60)

    comparison  = load_model_comparison()
    cv          = load_xgb_cv()
    xgb_summary = load_xgb_summary()
    shap        = load_shap_importance()
    shortfall   = load_shortfall()
    anomalies   = load_anomalies()

    any_data = any(x is not None for x in [comparison, cv, xgb_summary, shap, shortfall])
    if not any_data:
        print("\n[ERROR] No pipeline output files found.")
        print("Run the full pipeline first: make ingest etl train risk evaluate")
        sys.exit(1)

    # ── Generate \\newcommand definitions ─────────────────────────────────
    commands = generate_commands(comparison, cv, xgb_summary, shortfall, anomalies)
    cmd_path = REPORT_DIR / "results_auto.tex"
    cmd_path.write_text(commands, encoding="utf-8")
    print(f"  ✓ {cmd_path}  ({len(commands)} chars)")

    # ── Generate standalone table files ───────────────────────────────────
    if comparison is not None:
        table_path = TABLES_DIR / "model_comparison.tex"
        table_path.write_text(generate_model_comparison_table(comparison), encoding="utf-8")
        print(f"  ✓ {table_path}")
    else:
        print("  [SKIP] model_comparison.tex — run make evaluate first")

    if cv is not None:
        table_path = TABLES_DIR / "xgb_cv.tex"
        table_path.write_text(generate_cv_table(cv), encoding="utf-8")
        print(f"  ✓ {table_path}")
    else:
        print("  [SKIP] xgb_cv.tex — run python src/models/xgboost_model.py first")

    if shap is not None:
        table_path = TABLES_DIR / "shap_importance.tex"
        table_path.write_text(generate_shap_table(shap), encoding="utf-8")
        print(f"  ✓ {table_path}")
    else:
        print("  [SKIP] shap_importance.tex — run python src/models/xgboost_model.py first")

    if shortfall is not None:
        table_path = TABLES_DIR / "coverage.tex"
        table_path.write_text(
            generate_coverage_table(shortfall, xgb_summary), encoding="utf-8"
        )
        print(f"  ✓ {table_path}")
    else:
        print("  [SKIP] coverage.tex — run python src/risk/shortfall.py first")

    print("=" * 60)
    print("Next step: cd report && pdflatex main.tex && pdflatex main.tex")
    print("=" * 60)


if __name__ == "__main__":
    main()
