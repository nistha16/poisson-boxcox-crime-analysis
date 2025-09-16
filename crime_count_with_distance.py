# crime_count_with_distance_demo.py
# Goal: Keep y = crime_count (proper count target) and give Box–Cox a real job
# by using a skewed continuous predictor: avg distance_to_station_km.
# Compare Poisson vs Linear, raw vs Box–Cox on the predictor.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_poisson_deviance, r2_score
from sklearn.model_selection import train_test_split


# =========================================================
# 1) LOAD + CHECK
# ---------------------------------------------------------
# Expecting columns (from your earlier aggregation):
#   Borough, year, month, crime_count, distance_to_station_km
df = pd.read_csv("crime_distance_monthly.csv")

# Keep/rename the columns we need
needed = ["Borough", "year", "month", "crime_count", "distance_to_station_km"]
df = df[needed].dropna()

# Types + strictly-positive numeric for Box–Cox
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["crime_count"] = df["crime_count"].astype(float)
df["distance_to_station_km"] = df["distance_to_station_km"].astype(float).clip(lower=1e-6)

# Chronological split (train on 2022–2024, test on 2025)
train_df = df[df["year"] < 2025].sort_values(["year", "month"])
test_df  = df[df["year"] == 2025].sort_values(["year", "month"])



X_train = train_df[["Borough", "month", "distance_to_station_km"]]
y_train = train_df["crime_count"]
X_test  = test_df[["Borough", "month", "distance_to_station_km"]]
y_test  = test_df["crime_count"]

cat_features = ["Borough", "month"]       # treat month as categorical
num_features = ["distance_to_station_km"] # Box–Cox candidate

# =========================================================
# 2) PREPROCESSORS & MODELS
# ---------------------------------------------------------
onehot = OneHotEncoder(handle_unknown="ignore")

pre_raw = ColumnTransformer([
    ("cat", onehot, cat_features),
    ("num", "passthrough", num_features),
])

pre_boxcox = ColumnTransformer([
    ("cat", onehot, cat_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), num_features),
])

# Baselines
pipe_p_raw = Pipeline([("pre", pre_raw),    ("reg", PoissonRegressor(alpha=0.0, max_iter=2000))])
pipe_l_raw = Pipeline([("pre", pre_raw),    ("reg", LinearRegression())])

# Pipelines with Box–Cox
pipe_p_bc  = Pipeline([("pre", pre_boxcox), ("reg", PoissonRegressor(alpha=0.0, max_iter=2000))])
pipe_l_bc  = Pipeline([("pre", pre_boxcox), ("reg", LinearRegression())])

# =========================================================
# 3) FIT & EVALUATE
# ---------------------------------------------------------
def eval_poisson(name, pipe):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    dev = mean_poisson_deviance(y_test, pred)
    print(f"{name:<25} -> MSE={mse:.4f} | PoissonDev={dev:.4f}")
    return pred, mse, dev

def eval_linear(name, pipe):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2  = r2_score(y_test, pred)
    print(f"{name:<25} -> MSE={mse:.4f} | R^2={r2:.4f}")
    return pred, mse, r2

print("\n=== RESULTS (Target = crime_count; feature = distance_to_station_km) ===")
pred_p_raw, mse_p_raw, dev_p_raw = eval_poisson("Baseline Poisson (raw)", pipe_p_raw)
pred_p_bc,  mse_p_bc,  dev_p_bc  = eval_poisson("Pipeline A (BC + Poisson)", pipe_p_bc)
pred_l_raw, mse_l_raw, r2_l_raw  = eval_linear("Baseline Linear  (raw)", pipe_l_raw)
pred_l_bc,  mse_l_bc,  r2_l_bc   = eval_linear("Pipeline C (BC + Linear)",  pipe_l_bc)

# =========================================================
# 4) PLOTS (all in one figure)
# ---------------------------------------------------------
# a) Show how Box–Cox changes the predictor distribution
pt = PowerTransformer(method="box-cox", standardize=True)
dist_raw = X_train[["distance_to_station_km"]]
dist_bc = pt.fit_transform(dist_raw)

# b) Pred vs Actual helper
def scatter(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolor="k")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Actual crime_count")
    ax.set_ylabel("Predicted crime_count")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Row 1: histograms of the numeric feature
axes[0, 0].hist(dist_raw.values, bins=30, edgecolor="black")
axes[0, 0].set_title("distance_to_station_km (raw)")
axes[0, 0].set_xlabel("km"); axes[0, 0].set_ylabel("freq")

axes[0, 1].hist(dist_bc, bins=30, edgecolor="black", color="#f28e8e")
axes[0, 1].set_title("Box–Cox(distance)")
axes[0, 1].set_xlabel("transformed")

# Bar chart of MSEs for quick compare (Poisson + Linear)
labels = ["Poiss-RAW", "Poiss-BC", "Lin-RAW", "Lin-BC"]
mses   = [mse_p_raw,  mse_p_bc,   mse_l_raw,  mse_l_bc]
axes[0, 2].bar(labels, mses)
axes[0, 2].set_title("Test-set MSE (lower is better)")
axes[0, 2].tick_params(axis='x', rotation=20)

# Row 2: Pred vs Actual
scatter(axes[1, 0], y_test, pred_p_raw, "Baseline Poisson (raw)")
scatter(axes[1, 1], y_test, pred_p_bc,  "Pipeline A: BC + Poisson")
scatter(axes[1, 2], y_test, pred_l_raw, "Baseline Linear (raw)")

plt.tight_layout()
plt.show()

# =========================================================
# 5) SUMMARY
# ---------------------------------------------------------
print("\n=== SUMMARY (Test = 2025) ===")
print(f"Baseline Poisson (raw)   MSE={mse_p_raw:.4f} | PoissonDev={dev_p_raw:.4f}")
print(f"BC + Poisson (Pipeline A)MSE={mse_p_bc:.4f}  | PoissonDev={dev_p_bc:.4f}")
print(f"Baseline Linear (raw)    MSE={mse_l_raw:.4f} | R^2={r2_l_raw:.4f}")
print(f"BC + Linear (Pipeline C) MSE={mse_l_bc:.4f}  | R^2={r2_l_bc:.4f}")
