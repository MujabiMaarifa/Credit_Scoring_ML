import streamlit as st
import pandas as pd
import joblib

# === Load model and encoders ===
model = joblib.load('credit_score_model.pkl')

# Load label encoders for categorical variables
occupation_encoder = joblib.load('encoders/occupation_encoder.pkl')
credit_mix_encoder = joblib.load('encoders/credit_mix_encoder.pkl')
payment_min_encoder = joblib.load('encoders/payment_min_encoder.pkl')
payment_behaviour_encoder = joblib.load('encoders/payment_behaviour_encoder.pkl')

# === Streamlit UI ===
st.title("🏦 Credit Score Prediction App")
st.markdown("Enter customer details below to predict their credit score.")

# ==== User Inputs ====
age = st.number_input("Age", min_value=18, max_value=100, step=1)
occupation = st.selectbox("Occupation", occupation_encoder.classes_.tolist())
annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0)
num_credit_cards = st.number_input("Number of Credit Cards", min_value=0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
num_of_loans = st.number_input("Number of Loans", min_value=0)
delay_from_due_date = st.number_input("Average Delay from Due Date (days)", min_value=0)
num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0)
changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, step=100.0)
num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0)
credit_mix = st.selectbox("Credit Mix", credit_mix_encoder.classes_.tolist())
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)
credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, step=0.01)
payment_of_min_amount = st.selectbox("Payment of Minimum Amount", payment_min_encoder.classes_.tolist())
total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=100.0)
amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, step=100.0)
payment_behaviour = st.selectbox("Payment Behaviour", payment_behaviour_encoder.classes_.tolist())
monthly_balance = st.number_input("Monthly Balance", step=100.0)
credit_history_age_months = st.number_input("Credit History Age (in months)", min_value=0)

# ==== Assemble input data ====
input_data = pd.DataFrame({
    'Age': [age],
    'Occupation': [occupation],
    'Annual_Income': [annual_income],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_cards],
    'Interest_Rate': [interest_rate],
    'Num_of_Loan': [num_of_loans],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_of_delayed_payment],
    'Changed_Credit_Limit': [changed_credit_limit],
    'Num_Credit_Inquiries': [num_credit_inquiries],
    'Credit_Mix': [credit_mix],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_Utilization_Ratio': [credit_utilization_ratio],
    'Payment_of_Min_Amount': [payment_of_min_amount],
    'Total_EMI_per_month': [total_emi_per_month],
    'Amount_invested_monthly': [amount_invested_monthly],
    'Payment_Behaviour': [payment_behaviour],
    'Monthly_Balance': [monthly_balance],
    'Credit_History_Age_Months': [credit_history_age_months]
})

# ==== Encode categorical features ====
input_data['Occupation'] = occupation_encoder.transform(input_data['Occupation'])
input_data['Credit_Mix'] = credit_mix_encoder.transform(input_data['Credit_Mix'])
input_data['Payment_of_Min_Amount'] = payment_min_encoder.transform(input_data['Payment_of_Min_Amount'])
input_data['Payment_Behaviour'] = payment_behaviour_encoder.transform(input_data['Payment_Behaviour'])

# ==== Predict ====
if st.button("Predict Credit Score"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Credit Score: **{prediction[0]}**")
