"""
app.py — Streamlit dashboard for the Retail Promotion Impact project.

Run with:
    streamlit run app.py

The app has four sections:
  1. Overview — high-level sales trends and store comparisons
  2. Promotion Analysis — how promotions affect sales across store types
  3. Model Performance — comparing the three trained models
  4. Scenario Simulator — interactive tool to forecast revenue uplift

Make sure you've run train.py first so the models/ folder exists.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_data,
    build_features,
    get_feature_columns,
    plot_sales_over_time,
    plot_promo_uplift_by_store_type,
    plot_feature_importance,
    PALETTE,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Promo Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# custom CSS — just tidying up spacing and font sizes
st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .section-header {
        color: #1a5276;
        border-bottom: 2px solid #2980b9;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #2980b9;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data and models ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_everything():
    """
    Load the dataset and all trained models once, then cache the result.
    Streamlit reruns the whole script on each interaction, so caching
    here makes navigation between tabs feel instant.
    """
    df = load_data()
    df = build_features(df)

    models = {}
    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest":     "random_forest.pkl",
        "XGBoost":           "xgboost.pkl",
    }
    for name, fname in model_files.items():
        path = os.path.join("models", fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    scaler_path = os.path.join("models", "scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    results = {}
    results_path = os.path.join("models", "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    with open(os.path.join("models", "feature_cols.json")) as f:
        feature_cols = json.load(f)

    return df, models, scaler, results, feature_cols


# check models have been trained before trying to load them
if not os.path.exists("models/results.json"):
    st.error("Models not found. Please run `python train.py` first, then relaunch the app.")
    st.stop()

with st.spinner("Loading data and models..."):
    df, models, scaler, results, feature_cols = load_everything()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Retail Promo Dashboard")
st.sidebar.markdown("Analysing promotion impact on daily sales across Rossmann stores.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Promotion Analysis", "Model Performance", "Scenario Simulator"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)  \n"
    "**Models:** Linear Regression, Random Forest, XGBoost"
)


# ── PAGE 1: Overview ──────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("<h2 class='section-header'>Sales Overview</h2>", unsafe_allow_html=True)

    # top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stores",        f"{df['Store'].nunique():,}")
    col2.metric("Date Range",          f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
    col3.metric("Avg Daily Sales",     f"€{df['Sales'].mean():,.0f}")
    col4.metric("Promo Days (% total)", f"{df['Promo'].mean()*100:.1f}%")

    st.markdown("---")

    # sales over time for a selected store
    st.subheader("Daily Sales — Single Store View")
    st.markdown("Select a store to see its full sales history. Blue shading marks days when a promotion was running.")

    store_id = st.selectbox("Store", sorted(df["Store"].unique()), index=0)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    plot_sales_over_time(df, store_id=store_id, ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # monthly average sales heatmap
    st.subheader("Average Sales by Month and Day of Week")
    st.markdown("Useful for spotting seasonal patterns — darker = higher average sales.")

    pivot = (
        df.groupby(["Month", "DayOfWeek"])["Sales"]
          .mean()
          .unstack()
          .rename(columns={0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
    )
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    import seaborn as sns
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="Blues",
        ax=ax2, linewidths=0.3, cbar_kws={"label": "Avg Sales (€)"}
    )
    ax2.set_xlabel("Day of Week")
    ax2.set_ylabel("Month")
    ax2.set_title("Average Daily Sales by Month × Day of Week", color=PALETTE["primary"])
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ── PAGE 2: Promotion Analysis ────────────────────────────────────────────────

elif page == "Promotion Analysis":
    st.markdown("<h2 class='section-header'>Promotion Impact Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        "Promotions don't lift sales equally across all stores. "
        "This section breaks down how different store types and time periods respond."
    )

    # overall promo uplift stats
    promo_avg    = df[df["Promo"] == 1]["Sales"].mean()
    no_promo_avg = df[df["Promo"] == 0]["Sales"].mean()
    uplift_pct   = (promo_avg - no_promo_avg) / no_promo_avg * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sales — Promo Day",    f"€{promo_avg:,.0f}")
    col2.metric("Avg Sales — No Promo",     f"€{no_promo_avg:,.0f}")
    col3.metric("Average Uplift",           f"+{uplift_pct:.1f}%")

    st.markdown("---")

    st.subheader("Uplift by Store Type")
    fig = plot_promo_uplift_by_store_type(df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # uplift by day of week
    st.subheader("Promotional Uplift by Day of Week")
    st.markdown("Are promotions more effective on certain days?")

    dow_promo = (
        df.groupby(["DayOfWeek", "Promo"])["Sales"]
          .mean()
          .unstack()
          .rename(columns={0: "No Promo", 1: "Promo"})
    )
    dow_promo.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_promo["Uplift %"] = ((dow_promo["Promo"] - dow_promo["No Promo"]) / dow_promo["No Promo"] * 100).round(1)

    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    bars = ax3.bar(dow_promo.index, dow_promo["Uplift %"],
                   color=[PALETTE["secondary"] if v > 0 else PALETTE["accent"] for v in dow_promo["Uplift %"]],
                   alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("Uplift (%)")
    ax3.set_title("Avg Promotional Sales Uplift by Day of Week", color=PALETTE["primary"])
    ax3.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, dow_promo["Uplift %"]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("---")
    st.subheader("Uplift Breakdown Table")
    st.dataframe(
        dow_promo.style.format({"No Promo": "€{:,.0f}", "Promo": "€{:,.0f}", "Uplift %": "{:.1f}%"}),
        use_container_width=True
    )


# ── PAGE 3: Model Performance ─────────────────────────────────────────────────

elif page == "Model Performance":
    st.markdown("<h2 class='section-header'>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown(
        "Three models were trained on the same data and evaluated on a held-out test set "
        "(all data from June 2015 onwards). Lower is better for all metrics."
    )

    # results table
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ["Model", "MAE (€)", "RMSE (€)", "RMSPE"]
    results_df = results_df.sort_values("RMSPE")

    st.subheader("Evaluation Metrics — Test Set")
    st.dataframe(
        results_df.style
            .format({"MAE (€)": "€{:,.0f}", "RMSE (€)": "€{:,.0f}", "RMSPE": "{:.4f}"})
            .highlight_min(subset=["MAE (€)", "RMSE (€)", "RMSPE"], color="#d6eaf8"),
        use_container_width=True,
        hide_index=True,
    )

    # metric comparison bars
    st.markdown("---")
    st.subheader("MAE Comparison")
    fig4, ax4 = plt.subplots(figsize=(7, 3))
    models_list = list(results.keys())
    maes = [results[m]["MAE"] for m in models_list]
    colors = [PALETTE["secondary"] if m != min(results, key=lambda x: results[x]["MAE"])
              else PALETTE["primary"] for m in models_list]
    ax4.barh(models_list, maes, color=colors, alpha=0.85)
    for i, v in enumerate(maes):
        ax4.text(v + 50, i, f"€{v:,.0f}", va="center", fontsize=10)
    ax4.set_xlabel("Mean Absolute Error (€)")
    ax4.set_title("Model MAE Comparison (lower = better)", color=PALETTE["primary"])
    ax4.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # feature importance
    st.markdown("---")
    st.subheader("What drives sales? — Random Forest Feature Importance")
    st.markdown(
        "Feature importance shows which variables the model relies on most when making predictions. "
        "Rolling sales history and lag features typically dominate — past sales are the strongest "
        "predictor of future sales."
    )
    if "Random Forest" in models:
        fi_fig = plot_feature_importance(feature_cols, models["Random Forest"].feature_importances_)
        st.pyplot(fi_fig, use_container_width=True)
        plt.close(fi_fig)


# ── PAGE 4: Scenario Simulator ────────────────────────────────────────────────

elif page == "Scenario Simulator":
    st.markdown("<h2 class='section-header'>Promotional Scenario Simulator</h2>", unsafe_allow_html=True)
    st.markdown(
        "Adjust the inputs below to simulate expected daily sales under different "
        "promotional conditions. This is designed so non-technical stakeholders can "
        "explore 'what if' scenarios without touching any code."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Store & Promotion Settings")

        store_type  = st.selectbox("Store Type", ["A", "B", "C", "D"])
        assortment  = st.selectbox("Assortment Level", ["Basic (a)", "Extra (b)", "Extended (c)"])
        promo_on    = st.toggle("Promotion Active", value=True)
        promo2_on   = st.toggle("Promo2 (ongoing loyalty promo)", value=False)
        school_hol  = st.toggle("School Holiday", value=False)
        comp_dist   = st.slider("Nearest Competitor Distance (km)", 0, 25, 5) * 1000

        st.subheader("Date Settings")
        day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        month       = st.selectbox("Month", list(range(1, 13)), index=5)

    # build a feature row from the sidebar inputs
    dow_map   = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
    st_map    = {"A":0,"B":1,"C":2,"D":3}
    assort_map= {"Basic (a)":0,"Extra (b)":1,"Extended (c)":2}

    dow_val = dow_map[day_of_week]

    # use median rolling/lag values from the real data as sensible defaults
    median_lag1    = float(df["Sales_lag_1"].median())
    median_lag7    = float(df["Sales_lag_7"].median())
    median_lag14   = float(df["Sales_lag_14"].median())
    median_roll7   = float(df["Sales_rolling_mean_7"].median())
    median_roll30  = float(df["Sales_rolling_mean_30"].median())
    median_std7    = float(df["Sales_rolling_std_7"].median())
    median_std30   = float(df["Sales_rolling_std_30"].median())

    scenario_features = {
        "DayOfWeek":             dow_val,
        "DayOfMonth":            15,
        "WeekOfYear":            26,
        "Month":                 month,
        "Year":                  2015,
        "IsWeekend":             int(dow_val >= 5),
        "IsMonthStart":          0,
        "IsMonthEnd":            0,
        "Quarter":               (month - 1) // 3 + 1,
        "Promo":                 int(promo_on),
        "Promo2":                int(promo2_on),
        "StoreType":             st_map[store_type],
        "Assortment":            assort_map[assortment],
        "CompetitionDistance":   comp_dist,
        "Sales_lag_1":           median_lag1,
        "Sales_lag_7":           median_lag7,
        "Sales_lag_14":          median_lag14,
        "Sales_rolling_mean_7":  median_roll7,
        "Sales_rolling_mean_30": median_roll30,
        "Sales_rolling_std_7":   median_std7,
        "Sales_rolling_std_30":  median_std30,
        "SchoolHoliday":         int(school_hol),
    }

    X_scenario = pd.DataFrame([scenario_features])[feature_cols]

    # predict with all three models
    predictions = {}
    for name, model in models.items():
        if name == "Linear Regression":
            X_input = scaler.transform(X_scenario)
        else:
            X_input = X_scenario
        pred_log = model.predict(X_input)[0]
        predictions[name] = round(np.expm1(pred_log), 2)

    with col_right:
        st.subheader("Predicted Daily Sales")

        best_model = "XGBoost" if "XGBoost" in predictions else "Random Forest"
        best_pred  = predictions.get(best_model, 0)

        st.metric(
            label=f"Best Model Estimate ({best_model})",
            value=f"€{best_pred:,.0f}",
        )

        # show all three predictions
        for name, pred in predictions.items():
            st.metric(label=name, value=f"€{pred:,.0f}")

        # show the uplift if promo is on vs off
        st.markdown("---")
        st.subheader("Promotion Uplift Estimate")

        # predict the no-promo baseline
        no_promo_features = scenario_features.copy()
        no_promo_features["Promo"] = 0
        X_no_promo = pd.DataFrame([no_promo_features])[feature_cols]

        if best_model == "Linear Regression":
            X_np_input = scaler.transform(X_no_promo)
        else:
            X_np_input = X_no_promo

        no_promo_pred = np.expm1(models[best_model].predict(X_np_input)[0])
        uplift_abs = best_pred - no_promo_pred
        uplift_pct = (uplift_abs / no_promo_pred * 100) if no_promo_pred > 0 else 0

        col_a, col_b = st.columns(2)
        col_a.metric("Without Promotion", f"€{no_promo_pred:,.0f}")
        col_b.metric("Uplift from Promotion",
                     f"€{uplift_abs:+,.0f}",
                     delta=f"{uplift_pct:+.1f}%")

        st.caption(
            f"Estimates based on median historical sales patterns. "
            f"Actual results will vary by specific store and campaign execution."
        )
