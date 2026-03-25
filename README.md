# Retail Promo Dashboard

This project analyses retail sales patterns using the Rossmann Store Sales dataset and builds machine learning models to predict daily store sales. The aim was to explore how promotional activity and historical sales behaviour relate to future sales, and then present the results through a simple Streamlit dashboard.

I trained and compared three models: Linear Regression, Random Forest, and XGBoost. The best-performing model was XGBoost, which achieved an RMSPE of **0.1304**, compared with **0.2277** for the Linear Regression baseline. This represents a **42.7% reduction in RMSPE** over the baseline model.

The project was built as an end-to-end workflow covering data preparation, feature engineering, model training, evaluation, and dashboard development.

---

## Project Overview

The project uses the Rossmann retail dataset to predict daily sales at store level. Rather than using a random train-test split, I used a time-based split so that the evaluation would better reflect a real forecasting setting. I also created lag and rolling features to help the models capture short-term sales patterns over time.

The dashboard was built to make the results easier to explore, including model comparison, sales trend analysis, and simple promotion-based scenario testing.

---

## Models Used

The following models were trained and evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

## Results

| Model | MAE | RMSE | RMSPE |
|------|----:|-----:|------:|
| Linear Regression | 1,058 | 2,062 | 0.2277 |
| Random Forest | 665 | 954 | 0.1407 |
| XGBoost | 631 | 888 | 0.1304 |

**Best model:** XGBoost  
**Improvement over baseline:** 42.7% reduction in RMSPE compared with Linear Regression

---

## What the Results Show

The tree-based models performed much better than Linear Regression, which suggests that the relationship between sales and the input features was not purely linear. Random Forest gave a strong improvement over the baseline, but XGBoost achieved the best overall performance across all three metrics.

The gap between the baseline and the stronger models shows that feature engineering and non-linear modelling added clear value to the forecasting task.

---

## Method

The overall workflow was:

- Load and clean the Rossmann sales data
- Create time-based features
- Engineer lag and rolling-window features
- Split the data using time rather than random sampling
- Train and compare multiple regression models
- Evaluate using MAE, RMSE, and RMSPE
- Save model outputs for use in a Streamlit dashboard

### Why RMSPE?

RMSPE was used as an important evaluation metric because percentage-based error is useful in retail forecasting, especially when sales values vary across different stores and time periods.

---

## Project Structure

```text
retail-promo-dashboard/
│
├── app.py
├── train.py
├── utils.py
├── models/
├── README.md
└── requirements.txt
