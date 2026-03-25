"""
train.py — model training, evaluation, and saving.

Run this script once to train the models and save everything needed
by the Streamlit app. It takes about 3-5 minutes on a standard laptop.

Usage:
    python train.py

What it does:
    1. Loads and merges the Rossmann train + store data
    2. Engineers all features (lags, rolling stats, calendar vars)
    3. Trains a Linear Regression baseline
    4. Trains a Random Forest Regressor
    5. Trains an XGBoost Regressor
    6. Evaluates all three on a held-out test set
    7. Saves models, feature columns, and results to the models/ folder
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — needed when running as a script
import matplotlib.pyplot as plt

from utils import (
    load_data,
    build_features,
    get_feature_columns,
    rmspe, mae, rmse,
    plot_feature_importance,
    plot_predictions_vs_actual,
    PALETTE,
)


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = "data"
MODELS_DIR = "models"
PLOTS_DIR  = "plots"

# using a time-based split rather than random — this is important for time
# series data because we don't want to train on future data
SPLIT_DATE = "2015-06-01"

RANDOM_STATE = 42


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(
        train_path=os.path.join(DATA_DIR, "train.csv"),
        store_path=os.path.join(DATA_DIR, "store.csv"),
    )
    print(f"  Loaded {len(df):,} rows covering {df['Store'].nunique()} stores")

    # ── 2. Feature engineering ────────────────────────────────────────────
    print("Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns()

    # time-based train/test split
    train_df = df[df["Date"] <  SPLIT_DATE]
    test_df  = df[df["Date"] >= SPLIT_DATE]

    X_train = train_df[feature_cols]
    y_train = train_df["Sales"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["Sales"]

    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # log-transform the target — sales distributions are right-skewed and
    # models generally perform better in log space for this kind of data
    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    results = {}

    # ── 3. Linear Regression baseline ────────────────────────────────────
    print("\nTraining Linear Regression baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_log)
    lr_preds_log = lr.predict(X_test_scaled)
    lr_preds     = np.expm1(lr_preds_log)  # back to original scale

    results["Linear Regression"] = {
        "MAE":   round(mae(y_test, lr_preds), 2),
        "RMSE":  round(rmse(y_test, lr_preds), 2),
        "RMSPE": round(rmspe(y_test, lr_preds), 4),
    }
    print(f"  MAE: {results['Linear Regression']['MAE']:,.0f} | "
          f"RMSE: {results['Linear Regression']['RMSE']:,.0f} | "
          f"RMSPE: {results['Linear Regression']['RMSPE']:.4f}")

    # save scaler alongside the linear model so app.py can apply it
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "linear_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)

    # ── 4. Random Forest ──────────────────────────────────────────────────
    print("\nTraining Random Forest (this takes a couple of minutes)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,           # use all CPU cores
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train_log)
    rf_preds_log = rf.predict(X_test)
    rf_preds     = np.expm1(rf_preds_log)

    results["Random Forest"] = {
        "MAE":   round(mae(y_test, rf_preds), 2),
        "RMSE":  round(rmse(y_test, rf_preds), 2),
        "RMSPE": round(rmspe(y_test, rf_preds), 4),
    }
    print(f"  MAE: {results['Random Forest']['MAE']:,.0f} | "
          f"RMSE: {results['Random Forest']['RMSE']:,.0f} | "
          f"RMSPE: {results['Random Forest']['RMSPE']:.4f}")

    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # feature importance plot
    fi_fig = plot_feature_importance(feature_cols, rf.feature_importances_)
    fi_fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fi_fig)

    # predictions vs actual plot
    pva_fig = plot_predictions_vs_actual(y_test, rf_preds, model_name="Random Forest")
    pva_fig.savefig(os.path.join(PLOTS_DIR, "rf_predictions_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close(pva_fig)

    # ── 5. XGBoost ────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)],
        verbose=False,
    )
    xgb_preds_log = xgb_model.predict(X_test)
    xgb_preds     = np.expm1(xgb_preds_log)

    results["XGBoost"] = {
        "MAE":   round(mae(y_test, xgb_preds), 2),
        "RMSE":  round(rmse(y_test, xgb_preds), 2),
        "RMSPE": round(rmspe(y_test, xgb_preds), 4),
    }
    print(f"  MAE: {results['XGBoost']['MAE']:,.0f} | "
          f"RMSE: {results['XGBoost']['RMSE']:,.0f} | "
          f"RMSPE: {results['XGBoost']['RMSPE']:.4f}")

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    xgb_pva_fig = plot_predictions_vs_actual(y_test, xgb_preds, model_name="XGBoost")
    xgb_pva_fig.savefig(os.path.join(PLOTS_DIR, "xgb_predictions_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close(xgb_pva_fig)

    # ── 6. Save results and metadata ─────────────────────────────────────
    print("\nSaving results...")
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # save the feature column list so the app loads the exact same features
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ── 7. Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'RMSPE':>8}")
    print("-" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<22} {metrics['MAE']:>10,.0f} "
              f"{metrics['RMSE']:>10,.0f} {metrics['RMSPE']:>8.4f}")
    print("=" * 55)

    best = min(results, key=lambda m: results[m]["RMSPE"])
    lr_rmspe = results["Linear Regression"]["RMSPE"]
    best_rmspe = results[best]["RMSPE"]
    improvement = round((1 - best_rmspe / lr_rmspe) * 100, 1)
    print(f"\nBest model: {best}")
    print(f"Improvement over linear baseline: {improvement}% reduction in RMSPE")
    print(f"\nAll model files saved to: {MODELS_DIR}/")
    print("Done. You can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    main()
