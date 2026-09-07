import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Grocery Demand Forecasting", layout="wide")

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_parquet("features.parquet")
    with open("backtest_results.json") as f:
        metrics = json.load(f)
    return df, metrics

models = load_models()
df, metrics = load_data()

FEATURES = ["store", "item", "dow", "month", "is_holiday", "promo",
            "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7"]

st.title("📦 Grocery Demand Forecasting — Quantile Model")
st.caption(
    "A distributional (quantile) demand forecasting model trained on a synthetic multi-store, "
    "multi-SKU grocery dataset generated to mirror real retail patterns — weekly seasonality, "
    "promo lift, holiday spikes, and store-level trend — since the standard public grocery "
    "datasets (e.g. Kaggle's M5) sit behind a login this demo can't authenticate through. "
    "The forecasting and evaluation methodology is the point: rolling-origin backtesting, "
    "p10/p50/p90 quantile forecasts, and comparison against a seasonal-naive baseline."
)

with st.sidebar:
    st.header("Backtest Results")
    st.metric("LightGBM Quantile WMAPE", f"{metrics['lgbm_quantile']['wmape_mean']*100:.1f}%")
    st.metric("Seasonal-Naive Baseline WMAPE", f"{metrics['seasonal_naive_baseline']['wmape_mean']*100:.1f}%")
    st.metric("Improvement over Naive", f"{metrics['improvement_over_naive_pct']:.1f}%")
    st.metric("80% Interval Coverage (target 80%)", f"{metrics['lgbm_quantile']['p10_p90_interval_coverage_mean']*100:.1f}%")
    st.caption(f"Evaluated across {metrics['n_folds']} rolling 4-week test windows.")
    st.divider()
    st.subheader("Select a series")
    store = st.selectbox("Store", sorted(df["store"].unique()))
    item = st.selectbox("Item", sorted(df["item"].unique()))
    horizon = st.slider("Forecast horizon (days)", 7, 60, 28)

series = df[(df["store"] == store) & (df["item"] == item)].sort_values("date").reset_index(drop=True)
history = series.tail(90)

# Roll the model forward autoregressively for the chosen horizon
last_row = series.iloc[-1].copy()
future_rows = []
recent_units = list(series["units"].tail(28).values)
cur_date = series["date"].iloc[-1]

for h in range(horizon):
    cur_date = cur_date + pd.Timedelta(days=1)
    feat = {
        "store": store, "item": item,
        "dow": cur_date.dayofweek, "month": cur_date.month,
        "is_holiday": int((cur_date.month == 12 and cur_date.day in (24, 25, 31)) or
                          (cur_date.month == 11 and 22 <= cur_date.day <= 28)),
        "promo": 0,
        "lag_7": recent_units[-7] if len(recent_units) >= 7 else recent_units[-1],
        "lag_14": recent_units[-14] if len(recent_units) >= 14 else recent_units[-1],
        "lag_28": recent_units[-28] if len(recent_units) >= 28 else recent_units[-1],
        "roll_mean_7": np.mean(recent_units[-7:]),
        "roll_mean_28": np.mean(recent_units[-28:]),
        "roll_std_7": np.std(recent_units[-7:]),
    }
    X = pd.DataFrame([feat])[FEATURES]
    preds = {q: float(models[q].predict(X)[0]) for q in models}
    future_rows.append({"date": cur_date, **{f"p{int(q*100)}": v for q, v in preds.items()}})
    recent_units.append(preds[0.5])

future = pd.DataFrame(future_rows)

fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"], y=history["units"], name="Actual (history)",
                          line=dict(color="#1F3864")))
fig.add_trace(go.Scatter(x=future["date"], y=future["p50"], name="Forecast (p50)",
                          line=dict(color="#D97706", dash="dash")))
fig.add_trace(go.Scatter(x=future["date"], y=future["p90"], name="p90", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future["date"], y=future["p10"], name="80% Interval (p10–p90)",
                          line=dict(width=0), fill="tonexty", fillcolor="rgba(217,119,6,0.2)"))
fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast table")
st.dataframe(future.round(1), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Model: LightGBM gradient-boosted trees trained separately at the 10th, 50th, and 90th "
    "percentiles (quantile/pinball loss objective) — a distributional forecasting approach, "
    "not a single point estimate. Backtested with rolling-origin (walk-forward) validation, "
    "the standard approach for time-series to avoid leaking future information into training."
)
