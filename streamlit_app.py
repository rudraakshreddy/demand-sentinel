"""
Serving Layer — dashboard.py

Streamlit multi-page interactive dashboard for the Retail Demand
Forecasting & Inventory Risk Monitoring Platform.

Pages:
  1. Overview          — KPIs, dataset summary, pipeline health
  2. Forecast Explorer — Per item-store forecast vs. actual overlay
  3. Risk Monitor      — Heatmap of SRI, anomaly timeline
  4. Model Evaluation  — Metric comparison table, SHAP waterfall
  5. Data Quality      — Null heatmap, ingestion log, price anomalies

Run: streamlit run src/serving/dashboard.py --server.port 8501
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
LOGS_DIR      = Path("logs")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #7c3aed;
    }
    .risk-high   { color: #ef4444; font-weight: bold; }
    .risk-medium { color: #f59e0b; font-weight: bold; }
    .risk-low    { color: #10b981; font-weight: bold; }
    .stMetric > label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders (cached) ─────────────────────────────────────────────────────

from sqlalchemy import create_engine

def get_engine():
    s = st.secrets["connections"]["postgresql"]
    pwd = s["password"].replace("@", "%40")
    uri = f"postgresql+psycopg2://{s['username']}:{pwd}@{s['host']}:{s['port']}/{s['database']}"
    return create_engine(uri)



# ── Egress-optimised loaders (query pre-aggregated views, TTL=1h) ─────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_kpi_summary() -> dict | None:
    """Single-row KPI aggregates — replaces 853K-row fact_sales scan."""
    try:
        df = pd.read_sql("SELECT * FROM v_kpi_summary", get_engine())
        return df.iloc[0].to_dict() if len(df) > 0 else None
    except Exception as e:
        st.error(f"DB Error: {e}"); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_sales() -> pd.DataFrame | None:
    """Daily aggregated sales (date × cat × store) — ~3 KB vs 12 MB raw."""
    try:
        df = pd.read_sql("SELECT date, cat_id, dept_id, store_id, n_items, total_sales, avg_sales, n_records FROM v_daily_sales ORDER BY date", get_engine())
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Database error in load_sales: {e}"); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_xgb_forecasts() -> pd.DataFrame | None:
    try:
        df = pd.read_sql("SELECT item_id, store_id, date, forecast, lower_ci, upper_ci FROM v_forecast_xgb", get_engine())
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e: st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_arima_results() -> pd.DataFrame | None:
    try:
        df = pd.read_sql("SELECT item_id, store_id, date, forecast, lower_ci, upper_ci FROM v_forecast_sarima", get_engine())
        df["date"] = pd.to_datetime(df["date"])
        df["model"] = "SARIMA"
        return df
    except Exception as e: st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_prophet_results() -> pd.DataFrame | None:
    try:
        df = pd.read_sql("SELECT item_id, store_id, date, forecast, lower_ci, upper_ci FROM v_forecast_prophet", get_engine())
        df["date"] = pd.to_datetime(df["date"])
        df["model"] = "Prophet"
        return df
    except Exception as e: st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_volatility() -> pd.DataFrame | None:
    """Pre-bucketed regime distribution — tiny (3 rows) vs 853K-row scan."""
    try:
        df = pd.read_sql("SELECT regime, count FROM v_volatility_regimes", get_engine())
        return df
    except Exception as e:
        st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_shortfall() -> pd.DataFrame | None:
    """KPI aggregate + SRI latest snapshot only."""
    try:
        kpi = pd.read_sql("SELECT breach_rate, avg_sri, n_records FROM v_shortfall_kpi", get_engine())
        sri = pd.read_sql("SELECT item_id, store_id, date, sri, shortfall_breach FROM v_sri_latest", get_engine())
        sri["date"] = pd.to_datetime(sri["date"])
        # Attach breach_rate as a column so downstream code still works
        sri["breach_rate"] = float(kpi["breach_rate"].iloc[0]) if len(kpi) > 0 else None
        return sri
    except Exception as e:
        st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_anomalies() -> pd.DataFrame | None:
    """Only flagged rows (already filtered view)."""
    try:
        df = pd.read_sql("SELECT item_id, store_id, date, if_score, cv, z_score, sales FROM v_anomalies", get_engine())
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e: st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_anomaly_timeline() -> pd.DataFrame | None:
    """Date-level anomaly counts — tiny (n_dates rows) for timeline chart."""
    try:
        df = pd.read_sql("SELECT date, anomaly_count FROM v_anomaly_by_date", get_engine())
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e: st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_shap() -> pd.DataFrame | None:
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    try:
        df = pd.read_sql("SELECT model_name as \"Model\", mape, smape, mae, rmse, wrmsse, coverage_95, interval_width FROM model_evaluation_results", get_engine())
        return df
    except Exception as e: st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_sales_raw_sample() -> pd.DataFrame | None:
    """Fetch 5 000 raw rows from fact_sales for the Data Quality schema view.
    Egress: ~300 KB — negligible."""
    try:
        df = pd.read_sql(
            "SELECT f.item_id, f.store_id, f.date_id as date, f.sales, i.cat_id, i.dept_id "
            "FROM fact_sales f JOIN dim_item i ON f.item_id = i.item_id LIMIT 5000",
            get_engine()
        )
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Database error in load_sales_raw_sample: {e}"); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_actuals_for_series(item_id: str, store_id: str) -> pd.DataFrame | None:
    """Fetch actual sales for one item-store pair — only ~28 rows, ~1 KB."""
    try:
        df = pd.read_sql(
            "SELECT date_id as date, sales FROM fact_sales "
            "WHERE item_id = %(item)s AND store_id = %(store)s ORDER BY date_id",
            get_engine(), params={"item": item_id, "store": store_id}
        )
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f'DB Error: {e}'); return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_sales_null_stats() -> pd.DataFrame | None:
    """Compute null counts for ALL 853K rows entirely in SQL — returns 1 row.
    Zero egress: only a tiny JSON summary is transferred."""
    try:
        sql = """
            SELECT
                COUNT(*)                               AS total_rows,
                COUNT(*) - COUNT(f.item_id)            AS item_id_nulls,
                COUNT(*) - COUNT(f.store_id)           AS store_id_nulls,
                COUNT(*) - COUNT(f.date_id)            AS date_nulls,
                COUNT(*) - COUNT(f.sales)              AS sales_nulls,
                COUNT(*) - COUNT(i.cat_id)             AS cat_id_nulls,
                COUNT(*) - COUNT(i.dept_id)            AS dept_id_nulls
            FROM fact_sales f
            JOIN dim_item i ON f.item_id = i.item_id
        """
        row = pd.read_sql(sql, get_engine()).iloc[0]
        total = int(row["total_rows"])
        cols  = ["item_id", "store_id", "date", "sales", "cat_id", "dept_id"]
        nulls = [int(row[f"{c}_nulls"]) for c in ["item_id", "store_id", "date", "sales", "cat_id", "dept_id"]]
        dtypes = ["object", "object", "date", "float64", "object", "object"]
        return pd.DataFrame({
            "Column":     cols,
            "Null Count": nulls,
            "Null %":     [round(n / total * 100, 4) for n in nulls],
            "Dtype":      dtypes,
            "Total Rows": [total] * len(cols),
        })
    except Exception as e:
        st.error(f"Database error in load_sales_null_stats: {e}"); return None


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════


# ══ Navigation ════════════════════════════════════════════════════════════════
PAGE = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "📈 Forecast Explorer",
    "⚠️ Risk Monitor",
    "🔬 Model Evaluation",
    "🧹 Data Quality",
])

if PAGE == "🏠 Overview":
    st.title("📦 Retail Demand Forecasting & Inventory Risk Platform")
    st.caption("M5 Forecasting Competition Dataset · Walmart Stores USA")
    st.divider()

    kpi       = load_kpi_summary()
    anomalies = load_anomalies()
    comparison = load_model_comparison()
    shortfall = load_shortfall()
    sales     = load_sales()   # v_daily_sales — aggregated, tiny

    col1, col2, col3, col4, col5 = st.columns(5)

    if kpi is not None:
        col1.metric("Total SKU-Store Series", f"{int(kpi.get('sku_store_series', 0)):,}")
        col2.metric("Days of History",        f"{int(kpi.get('days_of_history', 0)):,}")
        col3.metric("Total Sales Records",    f"{int(kpi.get('total_records', 0)):,}")
    if anomalies is not None:
        n_anom = len(anomalies)
        total  = int(kpi.get("total_records", 1)) if kpi else 1
        col4.metric("Anomaly Flags",
                    f"{n_anom:,}",
                    delta=f"{n_anom/total*100:.2f}% of records",
                    delta_color="off")
    if shortfall is not None and "breach_rate" in shortfall.columns:
        breach_rate = shortfall["breach_rate"].iloc[0]
        if breach_rate is not None and not pd.isna(breach_rate):
            col5.metric("Shortfall Breach Rate",
                        f"{breach_rate:.2%}",
                        delta="Target: 2.50%",
                        delta_color="normal" if abs(breach_rate - 0.025) < 0.01 else "inverse")
        else:
            col5.metric("Shortfall Breach Rate", "N/A", delta="Pending pipeline run", delta_color="off")

    st.divider()

    if sales is not None:
        st.subheader("📅 Aggregate Daily Demand — All Stores")
        agg = sales.groupby("date")["total_sales"].sum().reset_index()
        fig = px.area(agg, x="date", y="total_sales",
                      title="Total Daily Sales Across All M5 Item-Store Pairs",
                      labels={"total_sales": "Units Sold", "date": "Date"},
                      color_discrete_sequence=["#7c3aed"])
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if sales is not None:
            by_cat = sales.groupby("cat_id")["total_sales"].sum().reset_index()
            fig2 = px.bar(by_cat, x="cat_id", y="total_sales",
                          title="Total Sales by Category",
                          color="total_sales",
                          color_continuous_scale="Viridis",
                          labels={"cat_id": "Category", "total_sales": "Units"})
            st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        if sales is not None:
            by_store = sales.groupby("store_id")["avg_sales"].mean().reset_index()
            fig3 = px.bar(by_store, x="store_id", y="avg_sales",
                          title="Mean Daily Sales by Store",
                          color="avg_sales",
                          color_continuous_scale="RdYlGn",
                          labels={"store_id": "Store", "avg_sales": "Mean Units/Day"})
            st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Forecast Explorer
# ═══════════════════════════════════════════════════════════════════════════════

elif PAGE == "📈 Forecast Explorer":
    st.title("📈 Demand Forecast Explorer")
    st.caption("Compare SARIMA · Prophet · XGBoost forecasts per item-store pair")
    st.divider()

    xgb     = load_xgb_forecasts()
    arima   = load_arima_results()
    prophet = load_prophet_results()

    # Derive item/store lists from the forecast data (which still has item/store grain)
    fc_ref = xgb if xgb is not None else arima if arima is not None else prophet
    if fc_ref is None:
        st.warning("No forecast data found. Run the pipeline first: `make ingest etl train`")
        st.stop()

    # Selectors
    col1, col2, col3 = st.columns(3)
    all_stores = sorted(fc_ref["store_id"].unique())
    selected_store = col1.selectbox("Store", all_stores, index=0)
    items_in_store = sorted(fc_ref[fc_ref["store_id"] == selected_store]["item_id"].unique())
    selected_item = col2.selectbox("Item", items_in_store, index=0)
    show_ci = col3.checkbox("Show 95% CI bands", value=True)

    # Fetch actuals for selected series only (~28 rows, ~1 KB)
    actuals = load_actuals_for_series(selected_item, selected_store)

    fig = go.Figure()

    # Actual sales line
    if actuals is not None and not actuals.empty:
        fig.add_trace(go.Scatter(
            x=actuals["date"], y=actuals["sales"],
            name="Actual", mode="lines",
            line=dict(color="#94a3b8", width=1.5)
        ))

    # XGBoost forecast
    if xgb is not None:
        xgb_sub = xgb[(xgb["item_id"] == selected_item) &
                      (xgb["store_id"] == selected_store)].sort_values("date")
        if not xgb_sub.empty:
            fig.add_trace(go.Scatter(
                x=xgb_sub["date"], y=xgb_sub["forecast"],
                name="XGBoost", mode="lines+markers",
                line=dict(color="#7c3aed", width=2)
            ))

    # SARIMA forecast + CI
    if arima is not None:
        ar_sub = arima[(arima["item_id"] == selected_item) &
                       (arima["store_id"] == selected_store)].sort_values("date")
        if not ar_sub.empty:
            fig.add_trace(go.Scatter(
                x=ar_sub["date"], y=ar_sub["forecast"],
                name="SARIMA", mode="lines+markers",
                line=dict(color="#f59e0b", width=2)
            ))
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=pd.concat([ar_sub["date"], ar_sub["date"].iloc[::-1]]),
                    y=pd.concat([ar_sub["upper_ci"], ar_sub["lower_ci"].iloc[::-1]]),
                    fill="toself", fillcolor="rgba(245,158,11,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="SARIMA 95% CI"
                ))

    # Prophet forecast + CI
    if prophet is not None:
        pr_sub = prophet[(prophet["item_id"] == selected_item) &
                          (prophet["store_id"] == selected_store)].sort_values("date")
        if not pr_sub.empty:
            fig.add_trace(go.Scatter(
                x=pr_sub["date"], y=pr_sub["forecast"],
                name="Prophet", mode="lines+markers",
                line=dict(color="#10b981", width=2)
            ))
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=pd.concat([pr_sub["date"], pr_sub["date"].iloc[::-1]]),
                    y=pd.concat([pr_sub["upper_ci"], pr_sub["lower_ci"].iloc[::-1]]),
                    fill="toself", fillcolor="rgba(16,185,129,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Prophet 95% CI"
                ))

    fig.update_layout(
        title=f"Demand Forecast: {selected_item} × {selected_store}",
        xaxis_title="Date", yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show which models have coverage for this series
    missing = []
    if arima is not None and arima[(arima["item_id"] == selected_item) & (arima["store_id"] == selected_store)].empty:
        missing.append("SARIMA")
    if prophet is not None and prophet[(prophet["item_id"] == selected_item) & (prophet["store_id"] == selected_store)].empty:
        missing.append("Prophet")
    if missing:
        st.info(f"ℹ️ **{' & '.join(missing)}** forecasts are not available for this series. "
                f"These models were fitted on a representative sample of 30 high-volume series only. "
                f"Try selecting items like **FOODS_3_090**, **FOODS_3_586**, or **HOUSEHOLD_1_023** for full model coverage.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Risk Monitor
# ═══════════════════════════════════════════════════════════════════════════════

elif PAGE == "⚠️ Risk Monitor":
    st.title("⚠️ Inventory Risk Monitor")
    st.caption("Rolling Volatility · Anomaly Flags · Stockout Risk Index (SRI)")
    st.divider()

    vol       = load_volatility()       # v_volatility_regimes — 3 rows
    anomalies = load_anomalies()        # v_anomalies — flagged rows only
    anom_timeline = load_anomaly_timeline()  # v_anomaly_by_date — tiny
    shortfall = load_shortfall()        # v_sri_latest — one row per item/store

    # ── SRI Heatmap ──────────────────────────────────────────────────────────
    if shortfall is not None and "sri" in shortfall.columns and len(shortfall) > 0:
        st.subheader("🔥 Stockout Risk Index (SRI) — Latest Snapshot")
        pivot = shortfall.pivot_table(
            index="item_id", columns="store_id", values="sri", aggfunc="mean"
        )
        top_items = shortfall.groupby("item_id")["sri"].mean().nlargest(50).index
        pivot_top = pivot.loc[pivot.index.isin(top_items)]

        fig_heat = px.imshow(
            pivot_top,
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=1,
            title="SRI Heatmap: Top-50 Highest-Risk Items × All Stores",
            labels={"color": "SRI"},
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Volatility regime distribution ──────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if vol is not None and len(vol) > 0:
            st.subheader("Volatility Regime Distribution")
            fig_pie = px.pie(
                vol, values="count", names="regime",
                color="regime",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                title="Demand Volatility Regimes Across All Series"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.subheader("Volatility Regime Distribution")
            st.info("Volatility Risk pipeline data not yet generated.")

    with col_b:
        if anom_timeline is not None:
            st.subheader("Anomaly Timeline")
            fig_anom = px.bar(
                anom_timeline, x="date", y="anomaly_count",
                title="Daily Isolation Forest Anomaly Flags",
                labels={"date": "Date", "anomaly_count": "Anomaly Count"},
                color_discrete_sequence=["#ef4444"]
            )
            st.plotly_chart(fig_anom, use_container_width=True)

    # ── Top anomalies table ──────────────────────────────────────────────────
    if anomalies is not None:
        st.subheader("Top 20 Highest-Score Anomalies")
        top_anoms = anomalies.nsmallest(20, "if_score")[
            ["item_id", "store_id", "date", "sales", "z_score", "cv", "if_score"]
        ].reset_index(drop=True)
        top_anoms["if_score"] = top_anoms["if_score"].round(4)
        top_anoms["z_score"]  = top_anoms["z_score"].round(3)
        top_anoms["cv"]       = top_anoms["cv"].round(3)
        st.dataframe(
            top_anoms.style
              .background_gradient(subset=["if_score"], cmap="RdYlGn_r")
              .background_gradient(subset=["z_score"], cmap="Oranges"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

elif PAGE == "🔬 Model Evaluation":
    st.title("🔬 Model Evaluation")
    st.caption("Walk-forward Backtest Results · All metrics on identical holdout windows")
    st.divider()

    comparison = load_model_comparison()
    shap       = load_shap()

    if comparison is None:
        st.warning("Run `make evaluate` to compute evaluation metrics.")
        st.stop()

    # ── Metric table ─────────────────────────────────────────────────────────
    st.subheader("Model Comparison Table")
    st.caption("All metrics computed on the same 28-day out-of-sample holdout (walk-forward)")

    display_cols = [c for c in ["Model", "mape", "smape", "mae", "rmse",
                                 "wrmsse", "coverage_95", "interval_width"]
                    if c in comparison.columns]
    # Deduplicate: keep the best (lowest MAPE) run per model
    disp = comparison[display_cols].copy()
    for c in disp.columns:
        if c != "Model":
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)
    disp = disp.sort_values("mape").drop_duplicates(subset=["Model"]).sort_values("mape").reset_index(drop=True)

    st.dataframe(
        disp.style.highlight_min(subset=[c for c in display_cols
                                         if c not in ["Model", "coverage_95", "wrmsse"]],
                                 color="#bbf7d0")
                  .highlight_max(subset=["coverage_95"] if "coverage_95" in display_cols else [],
                                 color="#bbf7d0")
                  .format(na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("**N/A** in `wrmsse` — requires full 5-year training history (not loaded in cloud evaluation). "
               "**N/A** in `coverage_95` / `interval_width` for XGBoost — point-forecast model, no confidence intervals.")

    # ── Metric bar charts ────────────────────────────────────────────────────
    metric_options = [c for c in ["mape", "rmse", "mae", "smape"]
                      if c in disp.columns and disp[c].notna().any()]
    selected_metric = st.selectbox("Select metric to visualise", metric_options)

    if selected_metric:
        fig_bar = px.bar(
            disp, x="Model", y=selected_metric,  # use deduplicated disp, not raw comparison
            color="Model",
            color_discrete_sequence=["#7c3aed", "#10b981", "#f59e0b"],
            title=f"{selected_metric.upper()} by Model (lower is better)",
            text=selected_metric,
        )
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── SHAP feature importance ───────────────────────────────────────────────
    if shap is not None:
        st.subheader("XGBoost SHAP Feature Importance")
        mean_abs_shap = shap.abs().mean().sort_values(ascending=False).head(20)
        fig_shap = px.bar(
            x=mean_abs_shap.values,
            y=mean_abs_shap.index,
            orientation="h",
            title="Top 20 Features by Mean |SHAP| Value",
            labels={"x": "Mean |SHAP|", "y": "Feature"},
            color=mean_abs_shap.values,
            color_continuous_scale="Purples",
        )
        fig_shap.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)

    # ── Coverage calibration ─────────────────────────────────────────────────
    shortfall = load_shortfall()
    if shortfall is not None:
        st.subheader("Shortfall Risk Calibration")
        # Use the pre-aggregated breach_rate from v_shortfall_kpi (full dataset)
        # NOT shortfall["shortfall_breach"].mean() which only covers latest snapshot
        breach_rate = shortfall["breach_rate"].iloc[0] if "breach_rate" in shortfall.columns and len(shortfall) > 0 else None

        if breach_rate is None or pd.isna(breach_rate):
            st.info("Shortfall Risk pipeline data not yet generated. Run `make risk` locally to push data to Supabase.")
        else:
            target = 0.025
            gauge_max = max(20.0, round(breach_rate * 100 * 1.5))  # auto-scale so needle never clips

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=breach_rate * 100,
                delta={"reference": target * 100, "valueformat": ".2f"},
                title={"text": "Shortfall Breach Rate (%) vs 2.5% Target"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "steps": [
                        {"range": [0, target * 100 * 0.6], "color": "#fef9c3"},
                        {"range": [target * 100 * 0.6, target * 100 * 1.4], "color": "#bbf7d0"},
                        {"range": [target * 100 * 1.4, gauge_max], "color": "#fecaca"},
                    ],
                    "threshold": {
                        "line": {"color": "green", "width": 4},
                        "thickness": 0.75,
                        "value": target * 100
                    },
                    "bar": {"color": "#7c3aed"},
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Data Quality
# ═══════════════════════════════════════════════════════════════════════════════

elif PAGE == "🧹 Data Quality":
    st.title("🧹 Data Quality Monitor")
    st.caption("Ingestion logs · Null analysis · Schema validation")
    st.divider()

    null_stats = load_sales_null_stats()   # 1 row result from SQL COUNT on all 853K rows
    sales_raw  = load_sales_raw_sample()   # 5K rows for histogram only

    if null_stats is not None:
        st.subheader("Column-Level Null Analysis")
        st.caption(f"Computed over all **{null_stats['Total Rows'].iloc[0]:,}** records via server-side SQL aggregation.")
        st.dataframe(
            null_stats[["Column", "Null Count", "Null %", "Dtype"]].style
                .background_gradient(subset=["Null %"], cmap="Reds", vmin=0, vmax=100)
                .format({"Null %": "{:.4f}%"}),
            use_container_width=True, hide_index=True,
        )

    if sales_raw is not None:
        st.subheader("Sales Distribution (Log Scale)")
        fig_hist = px.histogram(
            sales_raw[sales_raw["sales"] > 0], x="sales",
            nbins=100, log_y=True,
            title="Sales Distribution (excluding zero-sales days)",
            labels={"sales": "Units Sold"},
            color_discrete_sequence=["#7c3aed"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Ingestion log
    log_path = LOGS_DIR / "ingestion.log"
    if log_path.exists():
        st.subheader("Ingestion Log (Last 50 Lines)")
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-50:]
        st.code("".join(lines), language="text")

