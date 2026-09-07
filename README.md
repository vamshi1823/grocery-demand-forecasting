# Grocery Demand Forecasting — Quantile (Distributional) Model

A demand forecasting case study built around the same problem shape as
grocery/retail demand & labor planning: predicting *distributions* of future
demand (not just a point forecast), and evaluating that forecast on metrics
that matter operationally — WMAPE and prediction-interval coverage — rather
than accuracy alone.

## Data

The dataset (`grocery_demand_synthetic.csv`) is **synthetically generated**
(`train_and_backtest.py`, section 1) to mirror real grocery retail demand
patterns: 8 stores × 6 items × 2 years of daily data, with store-level trend,
weekly seasonality (weekend lift), monthly seasonality, promo lift, holiday
spikes, and per-series noise. Public grocery demand datasets such as Kaggle's
M5 competition data sit behind a login wall; this project generates a
comparable synthetic dataset instead so the modeling and evaluation pipeline
below can be built and run end-to-end without external credentials.

## Approach

- **Feature engineering**: day-of-week, month, holiday flag, promo flag,
  7/14/28-day lags, and 7/28-day rolling mean and std — all computed
  leak-free (shifted before rolling).
- **Model**: LightGBM gradient-boosted trees trained separately at the
  p10 / p50 / p90 quantiles (`objective="quantile"`), producing a full
  predictive distribution per series-day rather than a single number.
- **Baseline**: seasonal-naive (same weekday, 7 days prior) — the standard
  floor a demand model needs to beat.
- **Evaluation**: rolling-origin (walk-forward) backtesting across 4
  sequential 4-week test windows, scored on WMAPE (median forecast) and
  pinball loss (all three quantiles), plus p10–p90 interval coverage to
  check the model's uncertainty estimate is actually calibrated.

## Results (see `backtest_results.json` for the full run)

| Metric | LightGBM Quantile | Seasonal-Naive |
|---|---|---|
| WMAPE (mean across folds) | ~11.0% | ~17.4% |
| 80% interval coverage | ~78.4% (target: 80%) | — |
| Improvement over naive | **~37%** | — |

## Run it

```bash
pip install -r requirements.txt
python train_and_backtest.py   # regenerates data, retrains, rebacktests
streamlit run app.py           # interactive forecast explorer
```

## Files

- `train_and_backtest.py` — data generation, feature engineering, rolling-origin backtest, final model training
- `app.py` — Streamlit app: pick a store/item, see history + p10/p50/p90 forecast, view backtest metrics
- `backtest_results.json` — saved metrics from the last run
- `models.pkl` — trained quantile models (p10/p50/p90)
- `grocery_demand_synthetic.csv`, `features.parquet` — generated dataset and engineered features
