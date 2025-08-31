"""compare:
Baselines (raw features):
Poisson (Baseline‑P)
Linear (Baseline‑L)
With Box–Cox on the numeric feature (month):
Pipeline A: Two‑step Box–Cox → Poisson
Pipeline B: “oint Box–Cox+Poisson (inside pipeline)
Pipeline C: Two‑step Box–Cox → Linear
Pipeline D: Joint Box–Cox+Linear (inside pipeline)
Chronological split: train 2022–2024, test 2025"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_poisson_deviance, r2_score


# Expecting columns: Borough, year, month, crime_count
df = pd.read_csv("crime_counts_by_borough_year_month.csv").sort_values(["year", "month"]).reset_index(drop=True)

# Train on past (2022–2024), test on future (2025)
train_df = df[df["year"] < 2025].copy()
test_df  = df[df["year"] == 2025].copy()

X_train = train_df[["Borough", "month"]]
y_train = train_df["crime_count"].astype(float)
X_test  = test_df[["Borough", "month"]]
y_test  = test_df["crime_count"].astype(float)

categorical_features = ["Borough"]
numeric_features = ["month"]

enc = OneHotEncoder(handle_unknown="ignore")
boxcox = PowerTransformer(method="box-cox", standardize=True)

def print_metrics(title, y_true, y_pred, is_poisson):
    mse = mean_squared_error(y_true, y_pred)
    if is_poisson:
        dev = mean_poisson_deviance(y_true, y_pred)
        print(f"{title:<28} -> MSE={mse:.4f} | PoissonDev={dev:.4f}")
        return mse, dev
    else:
        r2 = r2_score(y_true, y_pred)
        print(f"{title:<28} -> MSE={mse:.4f} | R^2={r2:.4f}")
        return mse, r2

# 1. baselines(raw features)

pre_raw = ColumnTransformer([
    ("cat", enc, categorical_features),
    ("num", "passthrough", numeric_features),
])

baseline_poiss = Pipeline([
    ("pre", pre_raw),
    ("reg", PoissonRegressor(alpha=0.0, max_iter=1000)),
])
baseline_poiss.fit(X_train, y_train)
pred_baseline_poiss = baseline_poiss.predict(X_test)
m_baseline_poiss = print_metrics("Baseline Poisson (raw)", y_test, pred_baseline_poiss, is_poisson=True)

baseline_lin = Pipeline([
    ("pre", pre_raw),
    ("reg", LinearRegression()),
])
baseline_lin.fit(X_train, y_train)
pred_baseline_lin = baseline_lin.predict(X_test)
m_baseline_lin = print_metrics("Baseline Linear  (raw)", y_test, pred_baseline_lin, is_poisson=False)

# 2. pipelines with Box-Cox
# A: Two‑step (Box–Cox → Poisson)
pre_A = ColumnTransformer([
    ("cat", enc, categorical_features),
    ("num", boxcox, numeric_features),
])
pipe_A = Pipeline([
    ("pre", pre_A),
    ("reg", PoissonRegressor(alpha=0.0, max_iter=1000)),
])
pipe_A.fit(X_train, y_train)
pred_A = pipe_A.predict(X_test)
m_A = print_metrics("Pipeline A (BC + Poisson)", y_test, pred_A, is_poisson=True)

# B: “Joint” (Box–Cox inside pipeline) + Poisson
# NOTE?: (In sklearn this is practically identical to A, but included for completeness.)
pre_B = ColumnTransformer([
    ("cat", enc, categorical_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), numeric_features),
])
pipe_B = Pipeline([
    ("pre", pre_B),
    ("reg", PoissonRegressor(alpha=0.0, max_iter=1000)),
])
pipe_B.fit(X_train, y_train)
pred_B = pipe_B.predict(X_test)
m_B = print_metrics("Pipeline B (Joint Poisson)", y_test, pred_B, is_poisson=True)

# C: Two‑step (Box–Cox → Linear)
pre_C = ColumnTransformer([
    ("cat", enc, categorical_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), numeric_features),
])
pipe_C = Pipeline([
    ("pre", pre_C),
    ("reg", LinearRegression()),
])
pipe_C.fit(X_train, y_train)
pred_C = pipe_C.predict(X_test)
m_C = print_metrics("Pipeline C (BC + Linear)", y_test, pred_C, is_poisson=False)

# D: Joint (Box–Cox inside pipeline) + Linear
pre_D = ColumnTransformer([
    ("cat", enc, categorical_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), numeric_features),
])
pipe_D = Pipeline([
    ("pre", pre_D),
    ("reg", LinearRegression()),
])
pipe_D.fit(X_train, y_train)
pred_D = pipe_D.predict(X_test)
m_D = print_metrics("Pipeline D (Joint Linear)", y_test, pred_D, is_poisson=False)

# # 3. summary
# print("\n=== SUMMARY (Test = 2025) ===")
# print(f"Baseline Poisson (raw)      MSE={m_baseline_poiss[0]:.4f} | PoissonDev={m_baseline_poiss[1]:.4f}")
# print(f"Pipeline A (BC + Poisson)   MSE={m_A[0]:.4f} | PoissonDev={m_A[1]:.4f}")
# print(f"Pipeline B (Joint Poisson)  MSE={m_B[0]:.4f} | PoissonDev={m_B[1]:.4f}")
# print(f"Baseline Linear  (raw)      MSE={m_baseline_lin[0]:.4f} | R^2={m_baseline_lin[1]:.4f}")
# print(f"Pipeline C (BC + Linear)    MSE={m_C[0]:.4f} | R^2={m_C[1]:.4f}")
# print(f"Pipeline D (Joint Linear)   MSE={m_D[0]:.4f} | R^2={m_D[1]:.4f}")


# 4. figures
# Month histograms: raw vs Box–Cox (from Poisson A) vs Box–Cox (from Linear C)
month_df = pd.DataFrame({"month": X_train["month"].values})
month_raw_vals = month_df["month"].values
month_bc_A = pipe_A.named_steps["pre"].named_transformers_["num"].transform(month_df)
month_bc_C = pipe_C.named_steps["pre"].named_transformers_["num"].transform(month_df)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# --- Row 1: Histograms ---
axes[0, 0].hist(month_raw_vals, bins=12, edgecolor="black")
axes[0, 0].set_title("Raw month")
axes[0, 0].set_xlabel("month (1–12)")
axes[0, 0].set_ylabel("frequency")

axes[0, 1].hist(month_bc_A, bins=12, edgecolor="black")
axes[0, 1].set_title("Box–Cox(month) for A")
axes[0, 1].set_xlabel("transformed")

axes[0, 2].hist(month_bc_C, bins=12, edgecolor="black")
axes[0, 2].set_title("Box–Cox(month) for C")
axes[0, 2].set_xlabel("transformed")

# Empty panel to balance layout
axes[0, 3].axis("off")

# --- Row 2: Pred vs Actual (Baselines + Box–Cox versions) ---
def scatter(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolor="k")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Actual crime_count")
    ax.set_ylabel("Predicted crime_count")

scatter(axes[1, 0], y_test, pred_baseline_poiss, "Baseline Poisson (raw)")
scatter(axes[1, 1], y_test, pred_A,              "Pipeline A: BC + Poisson")
scatter(axes[1, 2], y_test, pred_baseline_lin,  "Baseline Linear (raw)")
scatter(axes[1, 3], y_test, pred_C,              "Pipeline C: BC + Linear")

plt.tight_layout()
plt.show()

