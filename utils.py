"""
utils.py — helper functions for data loading, feature engineering, and plotting.

I've kept everything modular here so the training script and the Streamlit app
can both import what they need without duplicating logic. If you want to tweak
how features are built (e.g. add more lag windows), this is the only file
you need to change.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(train_path="data/train.csv", store_path="data/store.csv"):
    """
    Load and merge the Rossmann train and store metadata files.
    Returns a single merged DataFrame, with some basic type cleanup applied.
    """
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path)

    df = train.merge(store, on="Store", how="left")

    # drop days when the store was closed — no point trying to forecast zero sales
    df = df[df["Open"] == 1].copy()
    df = df[df["Sales"] > 0].copy()

    # sort chronologically so lag features don't leak future information
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df):
    """
    Construct all model features from the raw merged DataFrame.

    Three broad categories:
      1. Calendar features — day, week, month, year, whether it's a weekend
      2. Lag features — sales from 1, 7, and 14 days ago per store
      3. Rolling features — 7-day and 30-day rolling mean/std per store

    Lag and rolling features are computed within each store group so
    we don't accidentally mix sales history across different stores.
    """
    df = df.copy()

    # ── 1. Calendar ──────────────────────────────────────────────────────
    df["DayOfWeek"]    = df["Date"].dt.dayofweek          # 0 = Monday
    df["DayOfMonth"]   = df["Date"].dt.day
    df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"]        = df["Date"].dt.month
    df["Year"]         = df["Date"].dt.year
    df["IsWeekend"]    = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"]   = df["Date"].dt.is_month_end.astype(int)

    # quarter — useful because German retail has strong seasonal patterns
    df["Quarter"] = df["Date"].dt.quarter

    # ── 2. Lag features (per store) ──────────────────────────────────────
    for lag in [1, 7, 14]:
        df[f"Sales_lag_{lag}"] = (
            df.groupby("Store")["Sales"]
              .shift(lag)
        )

    # ── 3. Rolling statistics (per store) ────────────────────────────────
    for window in [7, 30]:
        df[f"Sales_rolling_mean_{window}"] = (
            df.groupby("Store")["Sales"]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"Sales_rolling_std_{window}"] = (
            df.groupby("Store")["Sales"]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )

    # ── 4. Encode categoricals ───────────────────────────────────────────
    # StoreType and Assortment are single letters — map to integers
    df["StoreType"]  = df["StoreType"].map({"a": 0, "b": 1, "c": 2, "d": 3})
    df["Assortment"] = df["Assortment"].map({"a": 0, "b": 1, "c": 2})

    # fill any NaNs introduced by lagging (first rows per store)
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    df[lag_cols] = df[lag_cols].fillna(0)

    # also fill store metadata NaNs
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["Promo2"] = df["Promo2"].fillna(0)

    return df


def get_feature_columns():
    """
    Return the exact list of columns used as model inputs.
    Keeping this in one place means train.py and app.py always agree
    on which features to pass to the model.
    """
    return [
        # calendar
        "DayOfWeek", "DayOfMonth", "WeekOfYear", "Month", "Year",
        "IsWeekend", "IsMonthStart", "IsMonthEnd", "Quarter",
        # promotions / store
        "Promo", "Promo2", "StoreType", "Assortment",
        "CompetitionDistance",
        # lag
        "Sales_lag_1", "Sales_lag_7", "Sales_lag_14",
        # rolling
        "Sales_rolling_mean_7", "Sales_rolling_mean_30",
        "Sales_rolling_std_7",  "Sales_rolling_std_30",
        # state holiday as numeric
        "SchoolHoliday",
    ]


# ── Evaluation metrics ────────────────────────────────────────────────────────

def rmspe(y_true, y_pred):
    """
    Root Mean Squared Percentage Error — the official Rossmann competition metric.
    More intuitive than raw RMSE for sales data because it's scale-independent.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # avoid divide-by-zero on any zero-sales rows that snuck through
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


# ── Plotting helpers ──────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#1a5276",
    "secondary": "#2980b9",
    "accent":    "#e74c3c",
    "light":     "#d6eaf8",
    "grey":      "#95a5a6",
}


def plot_sales_over_time(df, store_id=1, ax=None):
    """
    Line chart of daily sales for a single store.
    Overlays promotion periods as a shaded background so you can see
    the uplift effect at a glance.
    """
    store_df = df[df["Store"] == store_id].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(store_df["Date"], store_df["Sales"],
            color=PALETTE["primary"], linewidth=0.8, alpha=0.9, label="Daily sales")

    # shade promo days
    promo_days = store_df[store_df["Promo"] == 1]
    for _, row in promo_days.iterrows():
        ax.axvspan(row["Date"], row["Date"] + pd.Timedelta(days=1),
                   color=PALETTE["light"], alpha=0.4, linewidth=0)

    ax.set_title(f"Store {store_id} — Daily Sales (shaded = promo days)",
                 fontsize=12, color=PALETTE["primary"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_promo_uplift_by_store_type(df):
    """
    Bar chart comparing average sales on promo vs non-promo days,
    broken down by store type. Good for showing that promotional
    uplift isn't uniform — some store types respond much better.
    """
    summary = (
        df.groupby(["StoreType", "Promo"])["Sales"]
          .mean()
          .reset_index()
    )
    summary["StoreType"] = summary["StoreType"].map({0: "A", 1: "B", 2: "C", 3: "D"})
    summary["Promo_Label"] = summary["Promo"].map({0: "No Promo", 1: "Promo"})

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE["grey"], PALETTE["secondary"]]

    for i, (promo_val, group) in enumerate(summary.groupby("Promo")):
        label = "Promo" if promo_val == 1 else "No Promo"
        x = np.arange(len(group))
        ax.bar(x + i * 0.35, group["Sales"], width=0.35,
               color=colors[i], label=label, alpha=0.85)

    ax.set_xticks(np.arange(4) + 0.175)
    ax.set_xticklabels(["Type A", "Type B", "Type C", "Type D"])
    ax.set_ylabel("Avg Daily Sales (€)")
    ax.set_title("Promotional Uplift by Store Type", fontsize=12, color=PALETTE["primary"])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    return fig


def plot_feature_importance(feature_names, importances, top_n=15):
    """
    Horizontal bar chart of the top N feature importances from the Random Forest.
    Sorted so the most important feature is at the top — easier to read than
    a vertical bar chart when feature names are long.
    """
    fi = pd.Series(importances, index=feature_names).nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(fi.index, fi.values, color=PALETTE["secondary"], alpha=0.85)
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=12, color=PALETTE["primary"])
    ax.spines[["top", "right"]].set_visible(False)

    return fig


def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", n_points=300):
    """
    Scatter plot of predicted vs actual sales for a random sample of test rows.
    Points close to the diagonal line = good predictions.
    I sample to keep the plot readable — 300 points is enough to see the pattern.
    """
    idx = np.random.choice(len(y_true), size=min(n_points, len(y_true)), replace=False)
    y_t = np.array(y_true)[idx]
    y_p = np.array(y_pred)[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_t, y_p, alpha=0.3, s=15, color=PALETTE["primary"])

    # perfect prediction line
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    ax.plot(lims, lims, color=PALETTE["accent"], linewidth=1.5, linestyle="--", label="Perfect prediction")

    ax.set_xlabel("Actual Sales (€)")
    ax.set_ylabel("Predicted Sales (€)")
    ax.set_title(f"{model_name} — Predicted vs Actual", fontsize=12, color=PALETTE["primary"])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    return fig
