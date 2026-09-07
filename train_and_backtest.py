"""
Grocery Demand & Labor Forecasting — quantile (distributional) demand model
with rolling-origin backtesting against a seasonal-naive baseline.

Dataset: synthetic multi-store, multi-SKU daily demand — generated here to
mirror real grocery retail patterns (weekly seasonality, promo lift, holiday
spikes, store-level trend, noise) since Kaggle's M5 dataset is gated behind
a login this environment can't authenticate through. Honestly labeled as
synthetic throughout the README and app.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json

rng = np.random.default_rng(42)

# ---------------- 1. Generate synthetic multi-store grocery demand ----------------
N_STORES = 8
N_ITEMS = 6
N_DAYS = 730  # 2 years
start = pd.Timestamp("2024-09-01")
dates = pd.date_range(start, periods=N_DAYS, freq="D")

rows = []
for store in range(N_STORES):
    store_base = rng.uniform(40, 120)          # store size effect
    store_trend = rng.uniform(-0.01, 0.03)      # slow drift per day
    for item in range(N_ITEMS):
        item_mult = rng.uniform(0.5, 2.0)
        weekly = rng.uniform(0.15, 0.35)        # weekend lift strength
        noise_scale = rng.uniform(0.08, 0.18)
        promo_days = rng.choice(N_DAYS, size=int(N_DAYS * 0.06), replace=False)
        promo_flags = np.zeros(N_DAYS, dtype=int)
        promo_flags[promo_days] = 1

        for i, d in enumerate(dates):
            dow = d.dayofweek
            month = d.month
            is_holiday = int((month == 12 and d.day in (24, 25, 31)) or (month == 11 and 22 <= d.day <= 28))
            weekend_lift = weekly * store_base if dow >= 5 else 0
            holiday_lift = 0.6 * store_base if is_holiday else 0
            promo_lift = 0.45 * store_base * promo_flags[i]
            trend = store_base * store_trend * (i / 365)
            seasonal_month = 0.12 * store_base * np.sin(2 * np.pi * month / 12)
            base_demand = (store_base + trend + weekend_lift + holiday_lift + promo_lift + seasonal_month) * item_mult
            noise = rng.normal(0, noise_scale * max(base_demand, 1))
            demand = max(0, base_demand + noise)
            rows.append((store, item, d, dow, month, is_holiday, promo_flags[i], demand))

df = pd.DataFrame(rows, columns=["store", "item", "date", "dow", "month", "is_holiday", "promo", "units"])
df["units"] = df["units"].round().astype(int)
df.to_csv("grocery_demand_synthetic.csv", index=False)
print("Generated:", df.shape, "rows across", N_STORES, "stores x", N_ITEMS, "items")

# ---------------- 2. Feature engineering ----------------
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

grp = df.groupby(["store", "item"])["units"]
df["lag_7"] = grp.shift(7)
df["lag_14"] = grp.shift(14)
df["lag_28"] = grp.shift(28)
df["roll_mean_7"] = grp.shift(1).groupby([df["store"], df["item"]]).rolling(7).mean().reset_index(drop=True)
df["roll_mean_28"] = grp.shift(1).groupby([df["store"], df["item"]]).rolling(28).mean().reset_index(drop=True)
df["roll_std_7"] = grp.shift(1).groupby([df["store"], df["item"]]).rolling(7).std().reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

FEATURES = ["store", "item", "dow", "month", "is_holiday", "promo",
            "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7"]
TARGET = "units"

# ---------------- 3. Rolling-origin backtest: quantile GBM vs seasonal-naive ----------------
QUANTILES = [0.1, 0.5, 0.9]
df = df.sort_values("date").reset_index(drop=True)
unique_dates = df["date"].unique()
N_FOLDS = 4
fold_size = 28  # 4-week rolling test windows
horizon_starts = unique_dates[-(N_FOLDS * fold_size):][::fold_size][:N_FOLDS]

def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

results = {"lgbm_quantile": {}, "seasonal_naive": {}}
pinball_scores = {q: [] for q in QUANTILES}
wmape_scores = []
coverage_scores = []
naive_wmape_scores = []

for fold_i, test_start in enumerate(horizon_starts):
    test_end = test_start + pd.Timedelta(days=fold_size)
    train = df[df["date"] < test_start]
    test = df[(df["date"] >= test_start) & (df["date"] < test_end)]
    if len(test) == 0 or len(train) == 0:
        continue

    preds = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=200,
            num_leaves=31, learning_rate=0.05, min_child_samples=20, verbosity=-1,
        )
        model.fit(train[FEATURES], train[TARGET])
        preds[q] = model.predict(test[FEATURES])
        pinball_scores[q].append(pinball_loss(test[TARGET].values, preds[q], q))

    median_pred = preds[0.5]
    wmape_scores.append(wmape(test[TARGET].values, median_pred))
    covered = (test[TARGET].values >= preds[0.1]) & (test[TARGET].values <= preds[0.9])
    coverage_scores.append(covered.mean())

    # seasonal-naive baseline: same weekday, 7 days back
    naive_pred = test["lag_7"].values
    naive_wmape_scores.append(wmape(test[TARGET].values, naive_pred))

    print(f"Fold {fold_i+1}: test window {test_start.date()} to {test_end.date()}, "
          f"n={len(test)}, LGBM WMAPE={wmape_scores[-1]:.3f}, "
          f"naive WMAPE={naive_wmape_scores[-1]:.3f}, 80% interval coverage={coverage_scores[-1]:.3f}")

summary = {
    "n_folds": len(wmape_scores),
    "lgbm_quantile": {
        "wmape_mean": float(np.mean(wmape_scores)),
        "pinball_loss_by_quantile": {str(q): float(np.mean(v)) for q, v in pinball_scores.items()},
        "p10_p90_interval_coverage_mean": float(np.mean(coverage_scores)),
    },
    "seasonal_naive_baseline": {
        "wmape_mean": float(np.mean(naive_wmape_scores)),
    },
    "improvement_over_naive_pct": float(
        (np.mean(naive_wmape_scores) - np.mean(wmape_scores)) / np.mean(naive_wmape_scores) * 100
    ),
}

with open("backtest_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))

# Train final full-data models (for the app to serve forecasts) and save
final_models = {}
for q in QUANTILES:
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200,
                           num_leaves=31, learning_rate=0.05, min_child_samples=20, verbosity=-1)
    m.fit(df[FEATURES], df[TARGET])
    final_models[q] = m

import pickle
with open("models.pkl", "wb") as f:
    pickle.dump(final_models, f)

df.to_parquet("features.parquet", index=False)
print("\nSaved: grocery_demand_synthetic.csv, backtest_results.json, models.pkl, features.parquet")
