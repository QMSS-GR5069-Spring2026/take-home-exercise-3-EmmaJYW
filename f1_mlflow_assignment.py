# Databricks notebook source
# MAGIC %md
# MAGIC # Take Homework #3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load and Explore the F1 Datasets

# COMMAND ----------

# MAGIC %pip install typing_extensions==4.7.1 mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

# List all available files in the F1 data volume
data_path = "/Volumes/gr5069/raw/f1_data"
files = os.listdir(data_path)
print("Available F1 datasets:")
for f in sorted(files):
    print(f"  - {f}")

# COMMAND ----------

import pandas as pd
import numpy as np

# Load the datasets we'll join together
results = pd.read_csv(f"{data_path}/results.csv")
races = pd.read_csv(f"{data_path}/races.csv")
drivers = pd.read_csv(f"{data_path}/drivers.csv")
constructors = pd.read_csv(f"{data_path}/constructors.csv")
qualifying = pd.read_csv(f"{data_path}/qualifying.csv")

print(f"results:      {results.shape}")
print(f"races:        {races.shape}")
print(f"drivers:      {drivers.shape}")
print(f"constructors: {constructors.shape}")
print(f"qualifying:   {qualifying.shape}")

# COMMAND ----------

# Quick look at each dataset
print("=== RESULTS columns ===")
print(results.columns.tolist())
print("\n=== RACES columns ===")
print(races.columns.tolist())
print("\n=== QUALIFYING columns ===")
print(qualifying.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Join Datasets and Feature Engineering
# MAGIC We join **results** with **races**, **drivers**, **constructors**, and **qualifying** to build a rich feature set.
# MAGIC
# MAGIC **Target variable:** `positionOrder` (final finishing position in a race)

# COMMAND ----------

# --- Join datasets ---
df = (
    results
    .merge(races[["raceId", "year", "round", "circuitId"]], on="raceId", how="left")
    .merge(drivers[["driverId", "dob", "nationality"]], on="driverId", how="left")
    .merge(constructors[["constructorId", "nationality"]], 
           on="constructorId", how="left", suffixes=("_driver", "_constructor"))
    .merge(qualifying[["raceId", "driverId", "position"]].rename(columns={"position": "qualifyingPosition"}),
           on=["raceId", "driverId"], how="left")
)

print(f"Joined dataset shape: {df.shape}")
df.head()

# COMMAND ----------

# --- Feature Engineering ---

# Convert grid and qualifying position to numeric
df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
df["qualifyingPosition"] = pd.to_numeric(df["qualifyingPosition"], errors="coerce")
df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")

# Calculate driver age at race time
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["driver_age"] = df["year"] - df["dob"].dt.year

# Encode categorical features
df["nationality_driver_encoded"] = df["nationality_driver"].astype("category").cat.codes
df["nationality_constructor_encoded"] = df["nationality_constructor"].astype("category").cat.codes

# Select features and target
feature_cols = [
    "grid",                          # Starting grid position
    "qualifyingPosition",            # Qualifying result
    "year",                          # Season year
    "round",                         # Round in the season
    "circuitId",                     # Which circuit
    "constructorId",                 # Which team
    "driver_age",                    # Driver age
    "nationality_driver_encoded",    # Driver nationality
    "nationality_constructor_encoded" # Constructor nationality
]

target = "positionOrder"

# Drop rows with missing values in our selected columns
df_model = df[feature_cols + [target]].dropna()

print(f"Final modeling dataset: {df_model.shape}")
print(f"\nFeature summary:")
df_model.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train/Test Split

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df_model[feature_cols]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Set Up MLflow Experiment and Logging Function

# COMMAND ----------

import mlflow
import mlflow.sklearn


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error
)
import matplotlib.pyplot as plt

# Create the MLflow experiment
mlflow.set_experiment("/Users/jw4853@columbia.edu/F1_Random_Forest_Experiment")

# COMMAND ----------

import mlflow
import mlflow.sklearn


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error
)
import matplotlib.pyplot as plt

# Create the MLflow experiment
mlflow.set_experiment("/Users/jw4853@columbia.edu/F1_Random_Forest_Experiment")

def train_and_log_model(params, X_train, X_test, y_train, y_test, run_name):
    """
    Train a Random Forest model and log everything to MLflow:
    - Hyperparameters
    - The model itself
    - All regression metrics
    - 2 artifacts (feature importance CSV + residuals plot)
    """
    with mlflow.start_run(run_name=run_name):
        
        # ---- 1. Log Hyperparameters ----
        mlflow.log_params(params)
        
        # ---- 2. Train the Model ----
        model = RandomForestRegressor(**params, oob_score=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # ---- 3. Calculate and Log ALL Metrics ----
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        med_ae = median_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        mlflow.log_metrics({
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "explained_variance": evs,
            "median_absolute_error": med_ae,
            "max_error": max_err,
            "mape": mape,
            "oob_score": model.oob_score_ 
        })
        
        # ---- 4. Log the Model ----
        mlflow.sklearn.log_model(model, "random-forest-model")
        
        # ---- 5. Artifact 1: Feature Importance CSV ----
        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        importance_path = f"/tmp/feature_importance_{run_name}.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature-importance")
        
        # ---- 6. Artifact 2: Residuals Plot ----
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Predicted Position")
        axes[0].set_ylabel("Residual")
        axes[0].set_title(f"Residuals Plot - {run_name}")
        
        # Predicted vs Actual
        axes[1].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                     color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Actual Position")
        axes[1].set_ylabel("Predicted Position")
        axes[1].set_title(f"Predicted vs Actual - {run_name}")
        
        plt.tight_layout()
        
        residuals_path = f"/tmp/residuals_{run_name}.png"
        fig.savefig(residuals_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(residuals_path, artifact_path="residuals")
        
        print(f"  {run_name}: R2={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}")
        
        return {
            "run_name": run_name,
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "explained_variance": evs,
            "params": params
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Run 10+ Experiments with Different Hyperparameters

# COMMAND ----------

model = RandomForestRegressor(
    **params,
    oob_score=True  # ← add this
)
# Define 12 unique hyperparameter combinations
param_grid = [
    {"n_estimators": 50,   "max_depth": 3,    "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
    {"n_estimators": 100,  "max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
    {"n_estimators": 100,  "max_depth": 10,   "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
    {"n_estimators": 200,  "max_depth": 5,    "min_samples_split": 5,  "min_samples_leaf": 2,  "random_state": 42},
    {"n_estimators": 200,  "max_depth": 10,   "min_samples_split": 5,  "min_samples_leaf": 2,  "random_state": 42},
    {"n_estimators": 200,  "max_depth": 15,   "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
    {"n_estimators": 500,  "max_depth": 5,    "min_samples_split": 10, "min_samples_leaf": 4,  "random_state": 42},
    {"n_estimators": 500,  "max_depth": 10,   "min_samples_split": 5,  "min_samples_leaf": 2,  "random_state": 42},
    {"n_estimators": 500,  "max_depth": 15,   "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
    {"n_estimators": 1000, "max_depth": 10,   "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
    {"n_estimators": 1000, "max_depth": 15,   "min_samples_split": 5,  "min_samples_leaf": 2,  "random_state": 42},
    {"n_estimators": 1000, "max_depth": None,  "min_samples_split": 2,  "min_samples_leaf": 1,  "random_state": 42},
]

# Run all experiments
all_results = []
print("Running 12 experiments...\n")

for i, params in enumerate(param_grid, 1):
    run_name = f"Run_{i}"
    result = train_and_log_model(params, X_train, X_test, y_train, y_test, run_name)
    all_results.append(result)

print("\nAll runs complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Compare All Runs and Select the Best Model

# COMMAND ----------

# Build comparison table
comparison = pd.DataFrame(all_results)

# Extract individual hyperparameters for display
comparison["n_estimators"] = comparison["params"].apply(lambda x: x["n_estimators"])
comparison["max_depth"] = comparison["params"].apply(lambda x: x["max_depth"])
comparison["min_samples_split"] = comparison["params"].apply(lambda x: x["min_samples_split"])
comparison["min_samples_leaf"] = comparison["params"].apply(lambda x: x["min_samples_leaf"])

display_cols = [
    "run_name", "n_estimators", "max_depth", 
    "min_samples_split", "min_samples_leaf",
    "r2", "mae", "rmse"
]

# Sort by R2 descending to see best models first
comparison_display = comparison[display_cols].sort_values("r2", ascending=False)
print("=== All Experiment Results (sorted by R²) ===\n")
print(comparison_display.to_string(index=False))

# COMMAND ----------

# Identify the best run
best = comparison.loc[comparison["r2"].idxmax()]

print(f"""
============================================
  BEST MODEL: {best['run_name']}
============================================

Hyperparameters:
  n_estimators:     {best['params']['n_estimators']}
  max_depth:        {best['params']['max_depth']}
  min_samples_split:{best['params']['min_samples_split']}
  min_samples_leaf: {best['params']['min_samples_leaf']}

Performance Metrics:
  R²:              {best['r2']:.4f}
  MAE:             {best['mae']:.2f}
  RMSE:            {best['rmse']:.2f}
  MSE:             {best['mse']:.2f}
  Explained Var:   {best['explained_variance']:.4f}
""")

depth_val = best['params']['max_depth']
if depth_val is None:
    depth_note = "unlimited depth (max_depth=None) — overfitting risk mitigated by ensemble averaging"
else:
    depth_note = f"constrained depth (max_depth={depth_val}) — good generalization signal"

print(f"Depth note for writeup: Best model used {depth_note}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Best Model Justification
# MAGIC
# MAGIC **Why this is the best model:**
# MAGIC
# MAGIC After running 12 experiments with varying hyperparameters (`n_estimators`, `max_depth`, 
# MAGIC `min_samples_split`, `min_samples_leaf`), the best model was selected based on the 
# MAGIC **highest R² score** (proportion of variance explained) combined with the **lowest MAE 
# MAGIC and RMSE** (prediction error in race positions).
# MAGIC
# MAGIC - **R² (primary metric):** The best run achieved the highest R², meaning it explains 
# MAGIC   the most variance in finishing position compared to all other configurations.
# MAGIC - **MAE / RMSE (secondary metrics):** It also produced the lowest prediction errors, 
# MAGIC   meaning the model's predicted finishing positions are closest to actual results.
# MAGIC - **Overfitting check:** Models with `max_depth=None` (unlimited depth) risk overfitting. 
# MAGIC   The best run used `max_depth=X` — [if constrained: "indicating the model generalizes 
# MAGIC   well without memorizing training data" / if None: "but Random Forest's ensemble 
# MAGIC   averaging across trees substantially reduces this risk compared to a single deep tree"].
# MAGIC   If the best model uses a constrained depth, this indicates good generalization. 
# MAGIC   If it uses unlimited depth but still performs best on the test set, 
# MAGIC   the ensemble averaging of Random Forest helps mitigate overfitting.
# MAGIC - **Complexity tradeoff:** Higher `n_estimators` improves stability but increases 
# MAGIC   training time. The selected model balances accuracy with computational cost.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Visualize Model Comparison

# COMMAND ----------

# Bar chart comparing R2 across all runs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

comparison_sorted = comparison.sort_values("r2", ascending=True)

# R² comparison
axes[0].barh(comparison_sorted["run_name"], comparison_sorted["r2"], color="steelblue")
axes[0].set_xlabel("R² Score")
axes[0].set_title("R² by Run (higher is better)")

# MAE comparison
axes[1].barh(comparison_sorted["run_name"], comparison_sorted["mae"], color="salmon")
axes[1].set_xlabel("MAE")
axes[1].set_title("MAE by Run (lower is better)")

# RMSE comparison
axes[2].barh(comparison_sorted["run_name"], comparison_sorted["rmse"], color="mediumseagreen")
axes[2].set_xlabel("RMSE")
axes[2].set_title("RMSE by Run (lower is better)")

plt.tight_layout()
plt.show()
