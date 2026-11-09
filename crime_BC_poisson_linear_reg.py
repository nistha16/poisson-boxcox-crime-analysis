import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_poisson_deviance
import matplotlib.pyplot as plt



# === 1. Load data ===
df = pd.read_csv("crime_density_by_borough_year_month.csv")

# Ensure correct types
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["crime_density"] = df["crime_density"].astype(float)  # y must be float
df["Area_km2"] = df["Area_km2"].astype(float)

# === 2. Chronological split (train 2022–2024, test 2025) ===
train_df = df[df["year"] < 2025].sort_values(["year", "month"])
test_df  = df[df["year"] == 2025].sort_values(["year", "month"])

X_train = train_df[["Borough", "month", "Area_km2"]]
y_train = train_df["crime_density"]
X_test  = test_df[["Borough", "month", "Area_km2"]]
y_test  = test_df["crime_density"]

# Categorical vs numeric
cat_features = ["Borough", "month"]  # month categorical
num_features = ["Area_km2"]

# === 3. Pipelines ===
# Preprocessor: encode categoricals, pass numeric unchanged
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", "passthrough", num_features)
])

# Preprocessor + Box–Cox
preprocessor_bc = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", PowerTransformer(method="box-cox"), num_features)  # BC on numeric only
])

# --- Baseline Poisson ---
pipe_pois = Pipeline([
    ("prep", preprocessor),
    ("model", PoissonRegressor(max_iter=500))
])

# --- Pipeline A: BC then Poisson ---
pipe_bc_pois = Pipeline([
    ("prep", preprocessor_bc),
    ("model", PoissonRegressor(max_iter=500))
])

# --- Pipeline B: Joint BC + Poisson ---
pipe_joint_pois = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", PowerTransformer(method="box-cox", standardize=True), num_features)
    ])),
    ("model", PoissonRegressor(max_iter=500))
])

# --- Baseline Linear ---
pipe_lin = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])

# --- Pipeline C: BC then Linear ---
pipe_bc_lin = Pipeline([
    ("prep", preprocessor_bc),
    ("model", LinearRegression())
])

# --- Pipeline D: Joint BC + Linear ---
pipe_joint_lin = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", PowerTransformer(method="box-cox", standardize=True), num_features)
    ])),
    ("model", LinearRegression())
])

# === 4. Fit & evaluate ===
def eval_model(name, pipe, metric="poisson"):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    if metric == "poisson":
        dev = mean_poisson_deviance(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        print(f"{name:<25} -> MSE={mse:.4f} | PoissonDev={dev:.4f}")
    else:
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        print(f"{name:<25} -> MSE={mse:.4f} | R^2={r2:.4f}")

print("\n=== RESULTS (Target = crime_density) ===")
eval_model("Baseline Poisson (raw)", pipe_pois, metric="poisson")
eval_model("Pipeline A (BC + Poisson)", pipe_bc_pois, metric="poisson")
eval_model("Pipeline B (Joint Poisson)", pipe_joint_pois, metric="poisson")
eval_model("Baseline Linear (raw)", pipe_lin, metric="linear")
eval_model("Pipeline C (BC + Linear)", pipe_bc_lin, metric="linear")
eval_model("Pipeline D (Joint Linear)", pipe_joint_lin, metric="linear")

# ==== 5. PLOTS ====
# Distribution before & after Box–Cox
pt = PowerTransformer(method="box-cox")
crime_density_bc = pt.fit_transform(train_df[["crime_density"]])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(train_df["crime_density"], bins=30, color='skyblue', edgecolor='black')
plt.title("Before Box–Cox (crime_density)")
plt.subplot(1,2,2)
plt.hist(crime_density_bc, bins=30, color='salmon', edgecolor='black')
plt.title("After Box–Cox (crime_density)")
plt.tight_layout()
plt.show()

# Predicted vs Actual
def plot_preds(y_true, y_pred, title):
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plot_preds(y_test, preds_baseline_poisson, "Baseline Poisson")
plt.subplot(2,2,2)
plot_preds(y_test, preds_A, "Pipeline A (BC + Poisson)")
plt.subplot(2,2,3)
plot_preds(y_test, preds_baseline_linear, "Baseline Linear")
plt.subplot(2,2,4)
plot_preds(y_test, preds_C, "Pipeline C (BC + Linear)")
plt.tight_layout()
plt.show()



"""Poisson models fit the crime density much better than linear models (MSE ~65 vs ~114 for raw).
Box–Cox (Pipeline A/B) reduced MSE from 65.23 → 40.39 — that’s a big drop, meaning transformation helped 
the Poisson regression capture the relationship better.
But notice: Poisson deviance got worse (0.86 → 1.73). That means while Box–Cox improved MSE (closer predictions
numerically), it moved away from the “natural” Poisson error
measure — possibly because the transformed predictor scale no longer matches the distribution assumption perfectly.
Linear models didn’t improve with Box–Cox — in fact, MSE and R² got slightly worse (probably because crime density 
was already quite linear with features).

So  conclusion could be:
“For crime density, Box–Cox transformation substantially reduced MSE for Poisson regression, indicating better 
numerical fit, but increased Poisson deviance, suggesting the transformed predictors may not fully align with the Poisson 
assumption. Linear regression did not benefit from the transformation.”"""