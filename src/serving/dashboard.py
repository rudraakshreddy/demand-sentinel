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

@st.cache_data(ttl=600, show_spinner=False)
def load_sales() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "sales_clean.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_xgb_forecasts() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "xgb_test_forecasts.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_arima_results() -> pd.DataFrame | None:
    import json
    p = PROCESSED_DIR / "arima_results.parquet"
    if not p.exists():
        return None
    raw = pd.read_parquet(p)
    rows = []
    for _, r in raw.iterrows():
        dates    = json.loads(r["dates"])
        forecast = json.loads(r["forecast"])
        actual   = json.loads(r["actual"])
        lower    = json.loads(r["lower_ci"])
        upper    = json.loads(r["upper_ci"])
        for d, f, a, lo, hi in zip(dates, forecast, actual, lower, upper):
            rows.append({
                "item_id": r["item_id"], "store_id": r["store_id"],
                "date": pd.to_datetime(d), "forecast": f, "actual": a,
                "lower_ci": lo, "upper_ci": hi, "model": "SARIMA",
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_prophet_results() -> pd.DataFrame | None:
    import json
    p = PROCESSED_DIR / "prophet_results.parquet"
    if not p.exists():
        return None
    raw = pd.read_parquet(p)
    rows = []
    for _, r in raw.iterrows():
        dates    = json.loads(r["dates"])
        forecast = json.loads(r["forecast"])
        actual   = json.loads(r["actual"])
        lower    = json.loads(r["lower_ci"])
        upper    = json.loads(r["upper_ci"])
        for d, f, a, lo, hi in zip(dates, forecast, actual, lower, upper):
            rows.append({
                "item_id": r["item_id"], "store_id": r["store_id"],
                "date": pd.to_datetime(d), "forecast": f, "actual": a,
                "lower_ci": lo, "upper_ci": hi, "model": "Prophet",
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_volatility() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "volatility.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_shortfall() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "shortfall_risk.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_anomalies() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "anomalies.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "model_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(ttl=600, show_spinner=False)
def load_shap() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "xgb_shap_values.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("📦 DFP Dashboard")
st.sidebar.caption("Retail Demand Forecasting Platform")
st.sidebar.divider()

PAGE = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Forecast Explorer", "⚠️ Risk Monitor",
     "🔬 Model Evaluation", "🧹 Data Quality"],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption("M5 Forecasting Dataset · PostgreSQL · XGBoost / Prophet / SARIMA")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if PAGE == "🏠 Overview":
    st.title("📦 Retail Demand Forecasting & Inventory Risk Platform")
    st.caption("M5 Forecasting Competition Dataset · Walmart Stores USA")
    st.divider()

    sales = load_sales()
    anomalies = load_anomalies()
    comparison = load_model_comparison()
    shortfall = load_shortfall()

    col1, col2, col3, col4, col5 = st.columns(5)

    if sales is not None:
        col1.metric("Total SKU-Store Series",
                    f"{sales.groupby(['item_id','store_id']).ngroups:,}")
        col2.metric("Days of History",
                    f"{sales['date'].nunique():,}")
        col3.metric("Total Sales Records",
                    f"{len(sales):,}")
    if anomalies is not None:
        col4.metric("Anomaly Flags",
                    f"{len(anomalies):,}",
                    delta=f"{len(anomalies)/len(sales)*100:.2f}% of records" if sales is not None else None,
                    delta_color="off")
    if shortfall is not None and "shortfall_breach" in shortfall.columns:
        breach_rate = shortfall["shortfall_breach"].mean()
        col5.metric("Shortfall Breach Rate",
                    f"{breach_rate:.2%}",
                    delta=f"Target: 5.00%",
                    delta_color="normal" if abs(breach_rate - 0.05) < 0.02 else "inverse")

    st.divider()

    if sales is not None:
        st.subheader("📅 Aggregate Daily Demand — All Stores")
        agg = sales.groupby("date")["sales"].sum().reset_index()
        fig = px.area(agg, x="date", y="sales",
                      title="Total Daily Sales Across All M5 Item-Store Pairs",
                      labels={"sales": "Units Sold", "date": "Date"},
                      color_discrete_sequence=["#7c3aed"])
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if sales is not None:
            by_cat = sales.groupby("cat_id")["sales"].sum().reset_index()
            fig2 = px.bar(by_cat, x="cat_id", y="sales",
                          title="Total Sales by Category",
                          color="sales",
                          color_continuous_scale="Viridis",
                          labels={"cat_id": "Category", "sales": "Units"})
            st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        if sales is not None:
            by_store = sales.groupby("store_id")["sales"].mean().reset_index()
            fig3 = px.bar(by_store, x="store_id", y="sales",
                          title="Mean Daily Sales by Store",
                          color="sales",
                          color_continuous_scale="RdYlGn",
                          labels={"store_id": "Store", "sales": "Mean Units/Day"})
            st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Forecast Explorer
# ═══════════════════════════════════════════════════════════════════════════════

elif PAGE == "📈 Forecast Explorer":
    st.title("📈 Demand Forecast Explorer")
    st.caption("Compare SARIMA · Prophet · XGBoost forecasts per item-store pair")
    st.divider()

    sales = load_sales()
    xgb   = load_xgb_forecasts()
    arima = load_arima_results()
    prophet = load_prophet_results()

    if sales is None:
        st.warning("Run the pipeline first: `make ingest etl train`")
        st.stop()

    # Selectors
    col1, col2, col3 = st.columns(3)
    all_items  = sorted(sales["item_id"].unique())
    all_stores = sorted(sales["store_id"].unique())

    selected_store = col1.selectbox("Store", all_stores, index=0)
    items_in_store = sorted(
        sales[sales["store_id"] == selected_store]["item_id"].unique()
    )
    selected_item = col2.selectbox("Item", items_in_store, index=0)
    show_ci = col3.checkbox("Show 95% CI bands", value=True)

    # Filter sales
    mask = (sales["item_id"] == selected_item) & (sales["store_id"] == selected_store)
    s = sales[mask].set_index("date")["sales"].sort_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, name="Actual", mode="lines",
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

    # Metric cards below
    if arima is not None and prophet is not None:
        ar_sub = arima[(arima["item_id"] == selected_item) & (arima["store_id"] == selected_store)]
        pr_sub = prophet[(prophet["item_id"] == selected_item) & (prophet["store_id"] == selected_store)]

        if not ar_sub.empty and not pr_sub.empty:
            from src.evaluation.metrics import mape, rmse
            st.subheader("Forecast Accuracy — This Series")
            c1, c2, c3 = st.columns(3)
            ar_mape = mape(ar_sub["actual"].values, ar_sub["forecast"].values)
            pr_mape = mape(pr_sub["actual"].values, pr_sub["forecast"].values)
            c1.metric("SARIMA MAPE",  f"{ar_mape:.2%}")
            c2.metric("Prophet MAPE", f"{pr_mape:.2%}")
            if xgb is not None:
                xgb_sub = xgb[(xgb["item_id"] == selected_item) & (xgb["store_id"] == selected_store)]
                if not xgb_sub.empty and "actual" in xgb_sub.columns:
                    xgb_mape = mape(xgb_sub["actual"].values, xgb_sub["forecast"].values)
                    c3.metric("XGBoost MAPE", f"{xgb_mape:.2%}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Risk Monitor
# ═══════════════════════════════════════════════════════════════════════════════

elif PAGE == "⚠️ Risk Monitor":
    st.title("⚠️ Inventory Risk Monitor")
    st.caption("Rolling Volatility · Anomaly Flags · Stockout Risk Index (SRI)")
    st.divider()

    vol       = load_volatility()
    anomalies = load_anomalies()
    shortfall = load_shortfall()

    if vol is None:
        st.warning("Run `make risk` to compute risk metrics first.")
        st.stop()

    # ── SRI Heatmap ──────────────────────────────────────────────────────────
    if shortfall is not None and "sri" in shortfall.columns:
        st.subheader("🔥 Stockout Risk Index (SRI) — Latest Snapshot")
        latest = shortfall.sort_values("date").groupby(
            ["item_id", "store_id"]
        ).last().reset_index()

        pivot = latest.pivot_table(
            index="item_id", columns="store_id", values="sri", aggfunc="mean"
        )
        # Show top-50 highest-risk items
        top_items = latest.groupby("item_id")["sri"].mean().nlargest(50).index
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
        if "volatility_regime" in vol.columns:
            st.subheader("Volatility Regime Distribution")
            regime_counts = vol["volatility_regime"].value_counts().reset_index()
            regime_counts.columns = ["Regime", "Count"]
            fig_pie = px.pie(
                regime_counts, values="Count", names="Regime",
                color="Regime",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                title="Demand Volatility Regimes Across All Series"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        if anomalies is not None:
            st.subheader("Anomaly Timeline")
            anom_by_date = anomalies.groupby("date").size().reset_index(name="count")
            fig_anom = px.bar(
                anom_by_date, x="date", y="count",
                title="Daily Isolation Forest Anomaly Flags",
                labels={"date": "Date", "count": "Anomaly Count"},
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
    disp = comparison[display_cols].copy()
    for c in disp.columns:
        if c != "Model":
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)

    st.dataframe(
        disp.style.highlight_min(subset=[c for c in display_cols
                                         if c not in ["Model", "coverage_95"]],
                                 color="#bbf7d0")
                  .highlight_max(subset=["coverage_95"] if "coverage_95" in display_cols else [],
                                 color="#bbf7d0"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Metric bar charts ────────────────────────────────────────────────────
    metric_options = [c for c in ["mape", "rmse", "mae", "smape", "wrmsse"]
                      if c in comparison.columns]
    selected_metric = st.selectbox("Select metric to visualise", metric_options)

    if selected_metric:
        fig_bar = px.bar(
            comparison, x="Model", y=selected_metric,
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
    if shortfall is not None and "shortfall_breach" in shortfall.columns:
        st.subheader("Shortfall Risk Calibration")
        breach_rate = shortfall["shortfall_breach"].mean()
        target = 0.05

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=breach_rate * 100,
            delta={"reference": target * 100, "valueformat": ".2f"},
            title={"text": "Shortfall Breach Rate (%) vs 5% Target"},
            gauge={
                "axis": {"range": [0, 15]},
                "steps": [
                    {"range": [0, 4], "color": "#fef9c3"},
                    {"range": [4, 6], "color": "#bbf7d0"},
                    {"range": [6, 15], "color": "#fecaca"},
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

    sales = load_sales()

    if sales is not None:
        st.subheader("Column-Level Null Analysis")
        null_df = pd.DataFrame({
            "Column":       sales.columns,
            "Null Count":   sales.isnull().sum().values,
            "Null %":       (sales.isnull().mean() * 100).round(2).values,
            "Dtype":        sales.dtypes.astype(str).values,
        })
        st.dataframe(
            null_df.style.background_gradient(subset=["Null %"], cmap="Reds"),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Sales Distribution (Log Scale)")
        fig_hist = px.histogram(
            sales[sales["sales"] > 0], x="sales",
            nbins=100, log_y=True,
            title="Sales Distribution (excluding zero-sales days)",
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
    else:
        st.info("Ingestion log not yet available. Run `make ingest`.")
