"""
Create lightweight pre-aggregated views in Supabase to slash egress.
Replaces full-table row scans with compact summaries.
"""
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres.nsgqnqhmhbkhocaofoyu:Jon_seeker%4001@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
engine = create_engine(DB_URL, pool_pre_ping=True)

VIEWS = {

    # 1. Daily aggregate sales (date × cat_id × store_id) — replaces 853K-row full scan
    "v_daily_sales": """
        CREATE OR REPLACE VIEW v_daily_sales AS
        SELECT
            f.date_id                          AS date,
            i.cat_id,
            i.dept_id,
            f.store_id,
            COUNT(DISTINCT f.item_id)          AS n_items,
            SUM(f.sales)                       AS total_sales,
            AVG(f.sales)                       AS avg_sales,
            COUNT(*)                           AS n_records
        FROM fact_sales f
        JOIN dim_item i ON f.item_id = i.item_id
        GROUP BY f.date_id, i.cat_id, i.dept_id, f.store_id
    """,

    # 2. KPI summary — tiny single-row aggregates for Overview metrics
    "v_kpi_summary": """
        CREATE OR REPLACE VIEW v_kpi_summary AS
        SELECT
            COUNT(DISTINCT CONCAT(f.item_id, '-', f.store_id)) AS sku_store_series,
            COUNT(DISTINCT f.date_id)                          AS days_of_history,
            COUNT(*)                                           AS total_records,
            SUM(f.sales)                                       AS total_units_sold
        FROM fact_sales f
    """,

    # 3. Anomaly summary (only flagged rows, already filtered) — still needed for Top-20 table
    "v_anomalies": """
        CREATE OR REPLACE VIEW v_anomalies AS
        SELECT
            r.item_id,
            r.store_id,
            r.date_id          AS date,
            r.anomaly_score    AS if_score,
            r.rolling_cv       AS cv,
            r.rolling_mean     AS z_score,
            f.sales
        FROM fact_risk_flags r
        LEFT JOIN fact_sales f
               ON f.item_id  = r.item_id
              AND f.store_id = r.store_id
              AND f.date_id  = r.date_id
        WHERE r.is_anomaly = true
          AND r.flag_type  = 'isolation_forest'
    """,

    # 4. Anomaly count per day (tiny) — for the timeline bar chart
    "v_anomaly_by_date": """
        CREATE OR REPLACE VIEW v_anomaly_by_date AS
        SELECT
            date_id   AS date,
            COUNT(*)  AS anomaly_count
        FROM fact_risk_flags
        WHERE is_anomaly = true
          AND flag_type  = 'isolation_forest'
        GROUP BY date_id
        ORDER BY date_id
    """,

    # 5. Volatility regime distribution (tiny) — for the pie chart
    "v_volatility_regimes": """
        CREATE OR REPLACE VIEW v_volatility_regimes AS
        SELECT
            CASE
                WHEN rolling_cv <= 0.5 THEN 'Low'
                WHEN rolling_cv <= 1.0 THEN 'Medium'
                ELSE 'High'
            END AS regime,
            COUNT(*) AS count
        FROM fact_risk_flags
        WHERE rolling_cv IS NOT NULL
        GROUP BY 1
    """,

    # 6. Shortfall KPI (single aggregate) — for Overview metric card
    "v_shortfall_kpi": """
        CREATE OR REPLACE VIEW v_shortfall_kpi AS
        SELECT
            AVG(shortfall_breach::int)  AS breach_rate,
            AVG(sri)                    AS avg_sri,
            COUNT(*)                    AS n_records
        FROM fact_shortfall
    """,

    # 7. SRI latest snapshot per item-store (for heatmap)
    "v_sri_latest": """
        CREATE OR REPLACE VIEW v_sri_latest AS
        SELECT DISTINCT ON (item_id, store_id)
            item_id,
            store_id,
            date,
            sri,
            shortfall_breach
        FROM fact_shortfall
        ORDER BY item_id, store_id, date DESC
    """,

    # 8. Forecast explorer per item-store (still needs to be filterable, keep lean columns)
    "v_forecast_xgb": """
        CREATE OR REPLACE VIEW v_forecast_xgb AS
        SELECT item_id, store_id, date_id AS date, forecast, lower_ci_95 AS lower_ci, upper_ci_95 AS upper_ci
        FROM fact_forecasts
        WHERE model_name = 'XGBoost'
    """,

    "v_forecast_sarima": """
        CREATE OR REPLACE VIEW v_forecast_sarima AS
        SELECT item_id, store_id, date_id AS date, forecast, lower_ci_95 AS lower_ci, upper_ci_95 AS upper_ci
        FROM fact_forecasts
        WHERE model_name = 'SARIMA'
    """,

    "v_forecast_prophet": """
        CREATE OR REPLACE VIEW v_forecast_prophet AS
        SELECT item_id, store_id, date_id AS date, forecast, lower_ci_95 AS lower_ci, upper_ci_95 AS upper_ci
        FROM fact_forecasts
        WHERE model_name = 'Prophet'
    """,
}

with engine.begin() as conn:
    for name, ddl in VIEWS.items():
        conn.execute(text(ddl))
        print(f"[OK] Created/replaced view: {name}")

print("\nAll views created successfully!")
