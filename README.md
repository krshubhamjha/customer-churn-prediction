# Customer Churn Prediction

A machine learning project to predict which telecom customers are likely to leave. Built this to practice end-to-end ML — from raw data to a deployed web app.

---

## What this project does

Takes customer data (tenure, monthly charges, contract type, services) and predicts whether that customer will churn or not. Also gives a churn probability score so the business can prioritise who to call first.

---

## Dataset

IBM Telco Customer Churn dataset — 7043 customers, 21 features. About 26% of customers churned which made it an imbalanced classification problem.

Found one interesting data issue during EDA — TotalCharges column looked clean but was actually stored as text (object dtype). Converting it to numeric revealed 11 hidden missing values which I filled with the median.

---

## What I found in EDA

Three things stood out clearly:

- Customers with tenure less than 12 months churn the most. Once someone stays past a year they rarely leave.
- Month-to-month contract customers churn way more than annual or 2-year contract customers. Makes sense — nothing locking them in.
- Churned customers pay higher monthly bills on average (~$80) vs loyal customers (~$65). High bill plus any service issue = they leave fast.

---

## Feature Engineering

Created 6 new features from existing columns based on the EDA findings:

- **charge_per_tenure** — monthly charges divided by tenure. High charge relative to short tenure = high risk.
- **total_services** — how many services the customer subscribes to. Fewer services = less locked in.
- **has_protection** — whether customer has online security or device protection.
- **is_new_customer** — simple flag for tenure less than 12 months.
- **avg_monthly_value** — total charges divided by tenure. More accurate than just monthly charges.
- **high_risk_flag** — customer pays over $65, tenure under 12 months, and on monthly contract. Direct combination of top 3 churn signals.

Interestingly after adding these features Logistic Regression outperformed XGBoost. The engineered features created linear relationships that LR could use directly while XGBoost had already found similar patterns on its own.

---

## Models tested

Trained 4 models and compared F1 score on the minority class (churners):

| Model | F1 Score | ROC AUC |
|-------|----------|---------|
| Logistic Regression | 0.6216 | 0.7526 |
| Random Forest | 0.5988 | 0.7310 |
| XGBoost | 0.5830 | 0.7204 |
| Decision Tree | 0.5096 | 0.6646 |

Used F1 instead of accuracy because accuracy is misleading on imbalanced data. A model that predicts "no churn" for everyone gets 74% accuracy but catches zero actual churners — completely useless.

Also applied SMOTE on training data only to handle the 26/74 class imbalance. Test data was kept as original to reflect real world distribution.

---

## Final model results

Logistic Regression with engineered features:

- Accuracy: 76.58%
- Precision: 54.42%
- Recall: 72.46%
- F1 Score: 62.16%
- ROC AUC: 75.26%

Recall of 72% means model catches about 3 out of every 4 actual churners which is good enough to be useful for a retention team.

---

## SHAP

Used SHAP to explain why the model flagged specific customers. Monthly charges came out as the strongest churn driver followed by contract type and charge per tenure. The waterfall plot for individual customers helps the retention team understand exactly what to address in their outreach call.

---

## Streamlit app

Built a simple web app where you can fill in customer details and get a churn probability score with risk level and retention recommendations.

To run locally:

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
├── notebooks/
│   ├── 01_EDA_Churn.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
├── app/
│   ├── streamlit_app.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── outputs/
│   └── plots/
├── data/
│   └── Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## Stack

Python, Pandas, Scikit-learn, XGBoost, SHAP, SMOTE, Streamlit, Plotly, Matplotlib, Seaborn

---

## About me

Shubham Kumar — Data Analyst with 2+ years in Oil & Gas and Industrial IoT.

- Email: shubhamjha12113@gmail.com
- LinkedIn: linkedin.com/in/shubhamjha99
