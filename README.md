# Customer Churn Prediction

[![Churn Prediction ML Pipeline](https://github.com/krshubhamjha/customer-churn-prediction/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/krshubhamjha/customer-churn-prediction/actions/workflows/ci_cd.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)

👉 **[Live Demo](https://customer-churn-prediction-bk6rqfd8h8rktvefudwnd4.streamlit.app/)**

---

A machine learning project to predict which telecom customers are likely to leave before they actually do. Started this to build something end to end — not just a notebook but a full system with automated retraining and deployment.

---

## The problem

A telecom company loses customers every month without knowing who is about to leave until it is too late. Winning back a churned customer costs 5x more than retaining one. The goal here was to build something that flags at-risk customers early enough for the retention team to act.

---

## Dataset

IBM Telco Customer Churn — 7043 customers, 21 features covering demographics, account info, and services subscribed. About 26% of customers churned making it an imbalanced classification problem.

One thing I noticed during EDA — TotalCharges looked clean but was stored as object dtype. Converting to numeric revealed 11 hidden missing values which I filled with the median. Small thing but the kind of issue that breaks a pipeline silently if you miss it.

---

## What the EDA showed

Three patterns stood out clearly:

- Customers with tenure under 12 months churn the most. Once someone stays past a year they rarely leave.
- Month-to-month contract customers churn at roughly 3x the rate of annual contract customers. Nothing locking them in.
- Churned customers pay higher monthly bills on average — around $80 vs $65 for loyal customers. High bill plus any service issue and they leave fast.

These three became the backbone of the feature engineering.

---

## Feature engineering

Created 6 new features directly from the EDA findings:

- **charge_per_tenure** — monthly charges divided by tenure. A customer paying $80 after 2 months is very different from one paying $80 after 5 years.
- **total_services** — count of services subscribed. Fewer services means less locked in and easier to leave.
- **has_protection** — whether customer has online security or device protection. More protection services means more perceived value.
- **is_new_customer** — simple flag for tenure under 12 months. Directly encodes the biggest churn signal from EDA.
- **avg_monthly_value** — total charges divided by tenure. Gives a truer picture of customer value than current monthly charges alone.
- **high_risk_flag** — customer pays over $65, tenure under 12 months, and on monthly contract. Combines the top 3 churn signals into one direct feature.

After adding these features Logistic Regression outperformed XGBoost — F1 went from 0.58 to 0.62. The engineered features created explicit linear relationships that LR could use directly. XGBoost already finds similar patterns internally so the benefit was smaller for it. Interesting to see a simpler model win after better feature engineering.

---

## Models tested

Trained 4 models and compared F1 score on the minority class since accuracy is misleading on imbalanced data:

| Model | F1 Score | ROC AUC |
|-------|----------|---------|
| Logistic Regression | 0.6216 | 0.7526 |
| Random Forest | 0.5988 | 0.7310 |
| XGBoost | 0.5830 | 0.7204 |
| Decision Tree | 0.5096 | 0.6646 |

Applied SMOTE on training data only to handle the class imbalance. Test data was kept at original distribution to reflect real world conditions.

Also did hyperparameter tuning with GridSearchCV on Logistic Regression — best params came back as C=1, penalty=l2, solver=saga which are essentially the defaults. Model was already at its ceiling for this dataset.

---

## Final model results

Logistic Regression with engineered features:

- Accuracy: 76.58%
- Precision: 54.42%
- Recall: 72.46%
- F1 Score: 62.16%
- ROC AUC: 75.26%

Recall of 72% means the model catches about 3 out of every 4 actual churners. The F1 of 0.62 is consistent with what others have achieved on this dataset — the ceiling is around 0.65 with available features. Getting higher would require richer data like customer complaint history or call center logs.

---

## SHAP explainability

Used SHAP to explain why the model flagged specific customers. Monthly charges came out as the strongest driver followed by charge_per_tenure and contract type. The waterfall plots help the retention team understand exactly what to address when calling a flagged customer — not just that they might churn but why.

---

## Automated ML pipeline

Built a production pipeline that runs automatically on every push to GitHub via GitHub Actions.

**Steps:**

1. `validate_data.py` — 8 quality checks on incoming data before anything else runs
2. `load_data.py` — loads original data, detects new_data.csv if present, combines both
3. `feature_engineering.py` — creates 6 features, encodes, scales, saves processed files
4. `train_model.py` — applies SMOTE, trains model, saves versioned model file
5. `evaluate_model.py` — compares new vs old F1 score, keeps the better model
6. `run_pipeline.py` — master script that runs all steps in order

**Adding new data:**

Rename your new CSV to `new_data.csv`, put it in the `data/` folder, and push to GitHub. The pipeline automatically detects it, combines with original data, retrains, and updates the model only if performance improves. Old model is never replaced unless the new one is provably better.

**Run locally:**

```bash
python src/run_pipeline.py
```

Pipeline also runs on a schedule — automatically on the 1st of every month.

---

## Streamlit app

Built a web app where you fill in customer details and get a churn probability with risk level and retention recommendations. Deployed free on Streamlit Cloud.

**Run locally:**

```bash
git clone https://github.com/krshubhamjha/customer-churn-prediction.git
cd customer-churn-prediction/app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Project structure

```
customer-churn-prediction/
├── .github/
│   └── workflows/
│       └── ci_cd.yml                  ← GitHub Actions pipeline
├── src/
│   ├── validate_data.py               ← Step 1: 8 data quality checks
│   ├── load_data.py                   ← Step 2: load and combine data
│   ├── feature_engineering.py         ← Step 3: create and encode features
│   ├── train_model.py                 ← Step 4: SMOTE + train model
│   ├── evaluate_model.py              ← Step 5: compare and keep best
│   └── run_pipeline.py                ← master pipeline runner
├── notebooks/
│   ├── 01_EDA_Churn.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
├── app/
│   ├── streamlit_app.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── data/
│   ├── Telco-Customer-Churn.csv       ← original data
│   └── new_data.csv                   ← drop new data here
├── metadata/
│   ├── pipeline_metadata.json
│   ├── training_results.json
│   └── evaluation_results.json
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── requirements.txt
└── README.md
```

---

## Stack

Python, Pandas, Scikit-learn, XGBoost, SHAP, SMOTE, Streamlit, Plotly, GitHub Actions

---

## Things I would do differently with more data

The F1 ceiling on this dataset is around 0.65 with available features. To push higher in a real setting I would look at adding customer complaint history, call center interaction logs, and app usage frequency. I would also replace the CSV-based pipeline with a proper database connection and use Airflow for scheduling instead of GitHub Actions cron.

---

## About me

Shubham Kumar — Data Analyst with 2+ years experience in Oil & Gas and Industrial IoT at Bosch and RNR Group.

- Email: shubhamjha12113@gmail.com
- LinkedIn: linkedin.com/in/shubhamjha99
- GitHub: github.com/krshubhamjha
