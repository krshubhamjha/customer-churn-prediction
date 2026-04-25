# ================================================================
# CUSTOMER CHURN PREDICTION — STREAMLIT APP
# Author: Shubham Kumar
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon  = "🔮",
    layout     = "wide"
)

# ----------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A237E;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-very-high {
        background: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-high {
        background: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-medium {
        background: #FFFDE7;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-low {
        background: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: #1A237E;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background: #283593;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        model    = joblib.load(os.path.join(base_path, 'best_model.pkl'))
        scaler   = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        features = joblib.load(os.path.join(base_path, 'feature_names.pkl'))
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler, feature_names = load_model()

# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------
st.markdown('<div class="main-header">🔮 Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict which customers are at risk of leaving — before they do.</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/customer-insight.png", width=80)
st.sidebar.title("📋 Customer Details")
st.sidebar.markdown("Fill in the customer information below:")

# Demographics
st.sidebar.subheader("👤 Demographics")
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents     = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Account Info
st.sidebar.subheader("📄 Account Information")
tenure          = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract        = st.sidebar.selectbox("Contract Type",
                  ["Month-to-month", "One year", "Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment         = st.sidebar.selectbox("Payment Method", [
                  "Electronic check",
                  "Mailed check",
                  "Bank transfer (automatic)",
                  "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges   = monthly_charges * tenure

# Services
st.sidebar.subheader("📡 Services")
phone_service    = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines",
                   ["No", "Yes", "No phone service"])
internet         = st.sidebar.selectbox("Internet Service",
                   ["DSL", "Fiber optic", "No"])
online_security  = st.sidebar.selectbox("Online Security",
                   ["No", "Yes", "No internet service"])
online_backup    = st.sidebar.selectbox("Online Backup",
                   ["No", "Yes", "No internet service"])
device_protect   = st.sidebar.selectbox("Device Protection",
                   ["No", "Yes", "No internet service"])
tech_support     = st.sidebar.selectbox("Tech Support",
                   ["No", "Yes", "No internet service"])
streaming_tv     = st.sidebar.selectbox("Streaming TV",
                   ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",
                   ["No", "Yes", "No internet service"])

# ----------------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------------
def prepare_input():
    data = {
        'gender'            : 1 if gender == 'Male' else 0,
        'SeniorCitizen'     : 1 if senior_citizen == 'Yes' else 0,
        'Partner'           : 1 if partner == 'Yes' else 0,
        'Dependents'        : 1 if dependents == 'Yes' else 0,
        'tenure'            : tenure,
        'PhoneService'      : 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling'  : 1 if paperless == 'Yes' else 0,
        'MonthlyCharges'    : monthly_charges,
        'TotalCharges'      : total_charges,
        'charge_per_tenure' : monthly_charges / (tenure + 1),
        'total_services'    : sum([
            phone_service    == 'Yes',
            online_security  == 'Yes',
            online_backup    == 'Yes',
            device_protect   == 'Yes',
            tech_support     == 'Yes',
            streaming_tv     == 'Yes',
            streaming_movies == 'Yes'
        ]),
        'has_protection'    : int(online_security == 'Yes') +
                              int(device_protect  == 'Yes'),
        'is_new_customer'   : 1 if tenure < 12 else 0,
        'avg_monthly_value' : total_charges / (tenure + 1),
        'high_risk_flag'    : int(
            monthly_charges > 65 and
            tenure < 12 and
            contract == 'Month-to-month'
        ),
        'MultipleLines_No phone service' :
            1 if multiple_lines   == 'No phone service'    else 0,
        'MultipleLines_Yes' :
            1 if multiple_lines   == 'Yes'                 else 0,
        'InternetService_Fiber optic' :
            1 if internet         == 'Fiber optic'         else 0,
        'InternetService_No' :
            1 if internet         == 'No'                  else 0,
        'OnlineSecurity_No internet service' :
            1 if online_security  == 'No internet service' else 0,
        'OnlineSecurity_Yes' :
            1 if online_security  == 'Yes'                 else 0,
        'OnlineBackup_No internet service' :
            1 if online_backup    == 'No internet service' else 0,
        'OnlineBackup_Yes' :
            1 if online_backup    == 'Yes'                 else 0,
        'DeviceProtection_No internet service' :
            1 if device_protect   == 'No internet service' else 0,
        'DeviceProtection_Yes' :
            1 if device_protect   == 'Yes'                 else 0,
        'TechSupport_No internet service' :
            1 if tech_support     == 'No internet service' else 0,
        'TechSupport_Yes' :
            1 if tech_support     == 'Yes'                 else 0,
        'StreamingTV_No internet service' :
            1 if streaming_tv     == 'No internet service' else 0,
        'StreamingTV_Yes' :
            1 if streaming_tv     == 'Yes'                 else 0,
        'StreamingMovies_No internet service' :
            1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes' :
            1 if streaming_movies == 'Yes'                 else 0,
        'Contract_One year' :
            1 if contract         == 'One year'            else 0,
        'Contract_Two year' :
            1 if contract         == 'Two year'            else 0,
        'PaymentMethod_Credit card (automatic)' :
            1 if payment == 'Credit card (automatic)'      else 0,
        'PaymentMethod_Electronic check' :
            1 if payment          == 'Electronic check'    else 0,
        'PaymentMethod_Mailed check' :
            1 if payment          == 'Mailed check'        else 0,
    }

    df_input             = pd.DataFrame([data])
    df_input             = df_input.reindex(columns=feature_names,
                                             fill_value=0)
    scale_cols           = ['tenure', 'MonthlyCharges', 'TotalCharges',
                             'charge_per_tenure', 'avg_monthly_value']
    scale_cols           = [c for c in scale_cols
                             if c in df_input.columns]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])
    return df_input

# ----------------------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------------------
predict_btn = st.sidebar.button("🔮 Predict Churn",
                                 use_container_width=True)

# ----------------------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------------------
if predict_btn:
    input_df    = prepare_input()
    probability = model.predict_proba(input_df)[0][1]
    prediction  = model.predict(input_df)[0]

    # Risk level
    if probability >= 0.70:
        risk_level = "🔴 VERY HIGH RISK"
        risk_color = "#F44336"
        risk_class = "risk-very-high"
        action     = "Urgent intervention needed!"
    elif probability >= 0.50:
        risk_level = "🟠 HIGH RISK"
        risk_color = "#FF9800"
        risk_class = "risk-high"
        action     = "Proactive outreach recommended"
    elif probability >= 0.30:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "#FFC107"
        risk_class = "risk-medium"
        action     = "Monitor closely"
    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "#4CAF50"
        risk_class = "risk-low"
        action     = "No action needed"

    # ---- Top metrics ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Probability",  f"{probability*100:.1f}%")
    col2.metric("Risk Level",          risk_level)
    col3.metric("Recommended Action",  action)
    col4.metric("Tenure",             f"{tenure} months")
    st.markdown("---")

    # ---- Gauge + Summary ----
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📊 Churn Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = probability * 100,
            title = {'text': "Churn Probability %",
                     'font': {'size': 16}},
            delta = {'reference': 50,
                     'increasing': {'color': '#F44336'},
                     'decreasing': {'color': '#4CAF50'}},
            gauge = {
                'axis'      : {'range'     : [0, 100],
                               'tickwidth' : 1},
                'bar'       : {'color'     : risk_color},
                'bgcolor'   : 'white',
                'steps'     : [
                    {'range': [0,  30],  'color': '#E8F5E9'},
                    {'range': [30, 50],  'color': '#FFFDE7'},
                    {'range': [50, 70],  'color': '#FFF3E0'},
                    {'range': [70, 100], 'color': '#FFEBEE'}
                ],
                'threshold' : {
                    'line' : {'color': 'red', 'width': 4},
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=320, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("👤 Customer Profile Summary")
        total_svc = sum([
            phone_service    == 'Yes',
            online_security  == 'Yes',
            online_backup    == 'Yes',
            device_protect   == 'Yes',
            tech_support     == 'Yes',
            streaming_tv     == 'Yes',
            streaming_movies == 'Yes'
        ])
        high_risk = (monthly_charges > 65 and
                     tenure < 12 and
                     contract == 'Month-to-month')

        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | 📅 Tenure | {tenure} months |
        | 💰 Monthly Charges | \${monthly_charges:.0f} |
        | 📋 Contract | {contract} |
        | 🌐 Internet | {internet} |
        | 🔧 Total Services | {total_svc} / 7 |
        | 🆕 New Customer | {'Yes ⚠️' if tenure < 12 else 'No ✅'} |
        | ⚠️ High Risk Flag | {'Yes 🔴' if high_risk else 'No 🟢'} |
        | 💳 Payment | {payment} |
        """)

    st.markdown("---")

    # ---- Business Recommendations ----
    st.subheader("💡 Business Recommendations")

    if probability >= 0.50:
        st.markdown(
            f'<div class="{risk_class}"><b>⚠️ Alert:</b> '
            f'This customer has <b>{probability*100:.1f}%</b> '
            f'chance of churning! Immediate action recommended.</div>',
            unsafe_allow_html=True
        )

        st.markdown("**Suggested retention actions:**")
        col7, col8 = st.columns(2)

        recommendations = []
        if contract == 'Month-to-month':
            recommendations.append(
                "📋 Offer discounted annual or 2-year contract")
        if monthly_charges > 65:
            recommendations.append(
                "💰 Offer loyalty discount on monthly bill")
        if tech_support in ['No']:
            recommendations.append(
                "🛠️ Add free tech support for 3 months")
        if tenure < 12:
            recommendations.append(
                "🎁 Offer new customer loyalty reward")
        if online_security in ['No']:
            recommendations.append(
                "🔒 Add free online security trial")
        if payment == 'Electronic check':
            recommendations.append(
                "💳 Encourage switch to auto-pay for discount")

        mid = len(recommendations) // 2
        with col7:
            for rec in recommendations[:mid+1]:
                st.warning(rec)
        with col8:
            for rec in recommendations[mid+1:]:
                st.warning(rec)

        # Revenue at risk
        st.markdown("---")
        st.subheader("💸 Revenue at Risk")
        annual_revenue = monthly_charges * 12
        col9, col10, col11 = st.columns(3)
        col9.metric("Monthly Revenue",  f"${monthly_charges:.0f}")
        col10.metric("Annual Revenue",  f"${annual_revenue:.0f}")
        col11.metric("Retention ROI",
                     f"${annual_revenue*0.1:.0f}",
                     help="Cost of 10% discount to retain customer")

    else:
        st.markdown(
            f'<div class="{risk_class}"><b>✅ Good News:</b> '
            f'This customer has only <b>{probability*100:.1f}%</b> '
            f'chance of churning.</div>',
            unsafe_allow_html=True
        )
        st.info("💡 Tip: Include in loyalty rewards program "
                "to maintain satisfaction and prevent future churn.")

else:
    # ---- Welcome Screen ----
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome to Customer Churn Predictor 👋

        This app predicts which telecom customers are likely to
        leave — so your retention team can act before it's too late.

        ### How to use:
        1. 👈 **Fill in customer details** in the left sidebar
        2. 🔮 **Click Predict Churn** button
        3. 📊 **See churn probability** and risk level
        4. 💡 **Get targeted recommendations** to retain customer

        ### About this model:
        Built using IBM Telco Customer Churn dataset with
        7,043 customers and 21 features.
        """)

    with col2:
        st.markdown("### 📈 Model Performance")
        st.metric("F1 Score",   "0.6216")
        st.metric("Accuracy",   "76.58%")
        st.metric("Recall",     "72.46%")
        st.metric("Precision",  "54.42%")
        st.metric("ROC AUC",    "0.7526")

    st.markdown("---")

    # Sample metrics
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  "7,043")
    c2.metric("Churn Rate",       "26.54%")
    c3.metric("Features Used",    "36+")
    c4.metric("Model",            "Logistic Regression")

    st.markdown("---")
    st.markdown("""
    ### 🔍 Key Churn Drivers Found in EDA:
    - **Tenure < 12 months** → New customers churn most
    - **Month-to-month contract** → Easiest to leave
    - **High monthly charges > \$65** → Price sensitive customers
    - **No tech support** → Feel less supported
    - **Fiber optic internet** → High expectations, easy to switch
    """)

    st.markdown("---")
    st.caption("Built by Shubham Kumar | "
               "IBM Telco Churn Dataset | "
               "Logistic Regression + Feature Engineering")
