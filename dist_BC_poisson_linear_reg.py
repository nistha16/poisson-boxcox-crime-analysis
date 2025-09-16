"""Predictors (X):
distance_to_station_km (continuous → Box–Cox candidate)
Borough (categorical → one‑hot)
month (categorical → one‑hot)
Target (y): crime_count
Split: Chronological — train = 2022–2024, test = 2025
Pipelines (6 total):
Baseline Poisson (raw)
A: Box–Cox → Poisson
B: “Joint” (Box–Cox inside pipeline) → Poisson
Baseline Linear (raw)
C: Box–Cox → Linear
D: “Joint” (Box–Cox inside pipeline) → Linear
Visuals (one figure):
Histograms of distance: raw vs Box–Cox (A) vs Box–Cox (C)
Bar chart comparing metrics for all 6 runs"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_poisson_deviance, r2_score
from sklearn.model_selection import train_test_split


# ==============================
# 1) LOAD + BASIC CLEAN
# ==============================
# Expecting columns: Borough, year, month, crime_count, distance_to_station_km
df = pd.read_csv("crime_distance_monthly.csv")

# Keep only needed columns and drop obvious bad rows
df = df[["Borough", "year", "month", "crime_count", "distance_to_station_km"]].copy()
df = df.dropna(subset=["Borough", "year", "month", "crime_count", "distance_to_station_km"])

# Ensure types
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["crime_count"] = df["crime_count"].astype(float)   # PoissonRegressor expects float y
# Box–Cox needs strictly positive numeric input
df["distance_to_station_km"] = df["distance_to_station_km"].clip(lower=1e-6)

# Chronological split: train on 2022–2024, test on 2025
train_df = df[df["year"] < 2025].sort_values(["year", "month"])
test_df  = df[df["year"] == 2025].sort_values(["year", "month"])

print("with square")
#  todo use distance isntainstead 
train_df["distance_sq"] = train_df["distance_to_station_km"] #** 2 
test_df["distance_sq"]  = test_df["distance_to_station_km"] #** 2

X_train = train_df[["Borough", "month", "distance_sq"]]
y_train = train_df["crime_count"]
X_test  = test_df[["Borough", "month", "distance_sq"]]
y_test  = test_df["crime_count"]


# Categorical vs numeric features
cat_features = ["Borough", "month"]   # treat month as categorical here
num_features = ["distance_sq"]

# Shared encoders/transformers
onehot = OneHotEncoder(handle_unknown="ignore")
boxcox = PowerTransformer(method="box-cox", standardize=True)

# Helper to print metrics
def metrics_poisson(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    dev = mean_poisson_deviance(y_true, y_pred)
    print(f"{name:<28} -> MSE={mse:.4f} | PoissonDev={dev:.4f}")
    return mse, dev

def metrics_linear(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{name:<28} -> MSE={mse:.4f} | R^2={r2:.4f}")
    return mse, r2

# ==============================
# 2) BASELINES (RAW FEATURES)
# ==============================
# Preprocess: one-hot for cats, pass distance as-is
pre_raw = ColumnTransformer([
    ("cat", onehot, cat_features),
    ("num", "passthrough", num_features),
])

# Baseline Poisson
pipe_p_raw = Pipeline([
    ("pre", pre_raw),
    ("reg", PoissonRegressor(alpha=0.0, max_iter=2000)),
])
pipe_p_raw.fit(X_train, y_train)
pred_p_raw = pipe_p_raw.predict(X_test)
m_p_raw = metrics_poisson("Baseline Poisson (raw)", y_test, pred_p_raw)

# Baseline Linear
pipe_l_raw = Pipeline([
    ("pre", pre_raw),
    ("reg", LinearRegression()),
])
pipe_l_raw.fit(X_train, y_train)
pred_l_raw = pipe_l_raw.predict(X_test)
m_l_raw = metrics_linear("Baseline Linear  (raw)", y_test, pred_l_raw)

# ==============================
# 3) BOX–COX PIPELINES
# ==============================
# A: Two‑step (Box–Cox → Poisson)
pre_A = ColumnTransformer([
    ("cat", onehot, cat_features),
    ("num", boxcox, num_features),
])
pipe_A = Pipeline([
    ("pre", pre_A),
    ("reg", PoissonRegressor(alpha=0.0, max_iter=2000)),
])
pipe_A.fit(X_train, y_train)
pred_A = pipe_A.predict(X_test)
m_A = metrics_poisson("Pipeline A (BC + Poisson)", y_test, pred_A)

# B: “Joint” (BC inside pipeline) + Poisson
# (Practically the same in sklearn; kept for symmetry with your design.)
pre_B = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), num_features),
])
pipe_B = Pipeline([
    ("pre", pre_B),
    ("reg", PoissonRegressor(alpha=0.0, max_iter=2000)),
])
pipe_B.fit(X_train, y_train)
pred_B = pipe_B.predict(X_test)
m_B = metrics_poisson("Pipeline B (Joint Poisson)", y_test, pred_B)

# C: Two‑step (Box–Cox → Linear)
pre_C = ColumnTransformer([
    ("cat", onehot, cat_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), num_features),
])
pipe_C = Pipeline([
    ("pre", pre_C),
    ("reg", LinearRegression()),
])
pipe_C.fit(X_train, y_train)
pred_C = pipe_C.predict(X_test)
m_C = metrics_linear("Pipeline C (BC + Linear)", y_test, pred_C)

# D: “Joint” (BC inside pipeline) + Linear
pre_D = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", PowerTransformer(method="box-cox", standardize=True), num_features),
])
pipe_D = Pipeline([
    ("pre", pre_D),
    ("reg", LinearRegression()),
])
pipe_D.fit(X_train, y_train)
pred_D = pipe_D.predict(X_test)
m_D = metrics_linear("Pipeline D (Joint Linear)", y_test, pred_D)

# ==============================
# 4) VISUALS — ONE FIGURE
# ==============================
# Pull raw distance (train) and transformed versions (A & C) for histograms
dist_df = pd.DataFrame({"distance_sq": X_train["distance_sq"].values})
# Use the fitted PTs from A and C:
pt_A = pipe_A.named_steps["pre"].named_transformers_["num"]
pt_C = pipe_C.named_steps["pre"].named_transformers_["num"]
dist_bc_A = pt_A.transform(dist_df)
dist_bc_C = pt_C.transform(dist_df)

# Bar chart metrics
labels = [
    "Poiss-RAW", "Poiss-BC(A)", "Poiss-BC(B)",
    "Lin-RAW",   "Lin-BC(C)",   "Lin-BC(D)"
]
mse_vals = [m_p_raw[0], m_A[0], m_B[0], m_l_raw[0], m_C[0], m_D[0]]

# Create figure with 2 rows x 3 cols:
# Row 1: Histograms (raw, A, C). Row 2: Bar chart of MSEs (all 6).
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Histograms
axes[0, 0].hist(dist_df, bins=30, edgecolor="black")
axes[0, 0].set_title("distance_to_station (raw)")
axes[0, 0].set_xlabel("km")
axes[0, 0].set_ylabel("frequency")

axes[0, 1].hist(dist_bc_A, bins=30, edgecolor="black")
axes[0, 1].set_title("Box–Cox distance (Poisson A)")
axes[0, 1].set_xlabel("transformed")

axes[0, 2].hist(dist_bc_C, bins=30, edgecolor="black")
axes[0, 2].set_title("Box–Cox distance (Linear C)")
axes[0, 2].set_xlabel("transformed")

# Bar chart of MSEs
axes[1, 0].bar(labels, mse_vals)
axes[1, 0].set_title("Test-set MSE (lower is better)")
axes[1, 0].tick_params(axis='x', rotation=25)
axes[1, 0].set_ylabel("MSE")

# Pred vs Actual for best Poisson + best Linear (visually concise)
# pick best by MSE within each family
poisson_preds = {"Poiss-RAW": pred_p_raw, "Poiss-BC(A)": pred_A, "Poiss-BC(B)": pred_B}
best_poiss_name = min(poisson_preds, key=lambda k: mean_squared_error(y_test, poisson_preds[k]))
best_poiss_pred = poisson_preds[best_poiss_name]

linear_preds = {"Lin-RAW": pred_l_raw, "Lin-BC(C)": pred_C, "Lin-BC(D)": pred_D}
best_lin_name = min(linear_preds, key=lambda k: mean_squared_error(y_test, linear_preds[k]))
best_lin_pred = linear_preds[best_lin_name]

def scatter(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolor="k")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Actual crime_count")
    ax.set_ylabel("Predicted crime_count")

scatter(axes[1, 1], y_test, best_poiss_pred, f"Best Poisson: {best_poiss_name}")
scatter(axes[1, 2], y_test, best_lin_pred,   f"Best Linear:  {best_lin_name}")

plt.tight_layout()
plt.show()

# ==============================
# 5) TEXT SUMMARY
# ==============================
print("\n=== SUMMARY (Test = 2025) ===")
print(f"Baseline Poisson (raw)      MSE={m_p_raw[0]:.4f} | PoissonDev={m_p_raw[1]:.4f}")
print(f"Pipeline A (BC + Poisson)   MSE={m_A[0]:.4f} | PoissonDev={m_A[1]:.4f}")
print(f"Pipeline B (Joint Poisson)  MSE={m_B[0]:.4f} | PoissonDev={m_B[1]:.4f}")
print(f"Baseline Linear  (raw)      MSE={m_l_raw[0]:.4f} | R^2={m_l_raw[1]:.4f}")
print(f"Pipeline C (BC + Linear)    MSE={m_C[0]:.4f} | R^2={m_C[1]:.4f}")
print(f"Pipeline D (Joint Linear)   MSE={m_D[0]:.4f} | R^2={m_D[1]:.4f}")



"""Baseline Poisson (raw)       -> MSE=254.2010 | PoissonDev=1.6334
Baseline Linear  (raw)       -> MSE=384.0386 | R^2=0.9901
Pipeline A (BC + Poisson)    -> MSE=259.7263 | PoissonDev=1.6450
Pipeline B (Joint Poisson)   -> MSE=259.7263 | PoissonDev=1.6450
Pipeline C (BC + Linear)     -> MSE=382.3315 | R^2=0.9902
Pipeline D (Joint Linear)    -> MSE=382.3315 | R^2=0.9902
What this means
Poisson beats Linear on this count target (lower MSE), which is expected.
Box–Cox on distance didn’t help here (A/B a bit worse than raw). Your top‑row histograms show that BC made distance more symmetric, but the predictive relationship with counts didn’t improve—so the raw distance worked just as well or better.
A and B identical: in scikit‑learn, putting PowerTransformer “outside” vs “inside” the pipeline is effectively the same; both estimate λ on the training set and then fit the Poisson model. So you’ll often see the same numbers.
What we can report
With a continuous predictor (distance_to_station_km), Poisson regression outperforms linear regression on held‑out 2025 data. Applying a Box–Cox transform to distance made the distribution more symmetric but did not improve test metrics; the raw distance gave the best Poisson deviance and MSE."""


"""Baseline Poisson (raw)       -> MSE=259.7955 | PoissonDev=1.6298
Baseline Linear  (raw)       -> MSE=380.8803 | R^2=0.9902
Pipeline A (BC + Poisson)    -> MSE=259.7347 | PoissonDev=1.6450
Pipeline B (Joint Poisson)   -> MSE=259.7347 | PoissonDev=1.6450
Pipeline C (BC + Linear)     -> MSE=382.3315 | R^2=0.9902
Pipeline D (Joint Linear)    -> MSE=382.3315 | R^2=0.9902
Poisson > Linear for count data (MSE is lower, Poisson deviance is meaningful).
Box–Cox here doesn’t drastically improve performance — which is itself a valid result, because it shows that transformation isn’t automatically “better”; it’s data-dependent.
If the transformed variable was more skewed and more predictive, the effect would be bigger."""