import streamlit as st
import pandas as pd
import joblib

# Load your trained model
# ==== Model Selection ====
st.markdown("🔍 Choose a Model for Prediction")
model_choice = st.selectbox(
    "Select Classification Model",
    ['Logistic Regression', 'Decision Tree', 'Random Forest']
)

# Map model names to filenames
model_files = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'Random Forest': 'random_forest_model.pkl'
}

# Load selected model
model_filename = model_files[model_choice]
model = joblib.load(model_filename)

st.title("🏦 Credit Score Prediction App")
st.markdown("Enter customer details below to predict their credit score.")

# ==== User Inputs ====

age = st.number_input("Age", min_value=18, max_value=100, step=1)
occupation = st.selectbox("Occupation", ['Engineer', 'Doctor', 'Teacher', 'Writer', 'Lawyer', 'Entrepreneur', 'Other'])
annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0)
num_credit_cards = st.number_input("Number of Credit Cards", min_value=0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
num_of_loans = st.number_input("Number of Loans", min_value=0)
delay_from_due_date = st.number_input("Average Delay from Due Date (days)", min_value=0)
num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0)
changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, step=100.0)
num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0)
credit_mix = st.selectbox("Credit Mix", ['Standard', 'Good', 'Bad'])
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)
credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, step=0.01)
payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ['Yes', 'No'])
total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=100.0)
amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, step=100.0)
payment_behaviour = st.selectbox("Payment Behaviour", ['Low_spent_Large_value_payments', 'High_spent_Medium_value_payments',
                                                        'Low_spent_Medium_value_payments', 'High_spent_Large_value_payments',
                                                        'High_spent_Small_value_payments', 'Low_spent_Small_value_payments'])
monthly_balance = st.number_input("Monthly Balance", step=100.0)
credit_history_age_months = st.number_input("Credit History Age (in months)", min_value=0)

# ==== Create input DataFrame ====
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

# ==== Load Preprocessing pickle files ====
occupation_encoder = joblib.load('occupation_encoder.pkl')
credit_mix_encoder = joblib.load('credit_mix_encoder.pkl')
payment_min_encoder = joblib.load('payment_of_min_amount_encoder.pkl')
payment_behaviour_encoder = joblib.load('payment_behaviour_encoder.pkl')
scaler = joblib.load('standard_scaler.pkl')
model_columns = joblib.load('model_columns.pkl')  # list of columns after one-hot encoding

# ==== Encode categorical variables using LabelEncoders ====
def safe_transform(encoder, column, value):
    if value not in encoder.classes_:
        st.error(f"The {value} is not known in '{column}'")
        st.stop()
    return encoder.transform([value])[0]
input_data['Occupation'] = safe_transform(occupation_encoder, 'Occupation', input_data['Occupation'].values[0])
input_data['Credit_Mix'] = safe_transform(credit_mix_encoder, 'Credit_Mix', input_data['Credit_Mix'])
input_data['Payment_of_Min_Amount'] = safe_transform(payment_min_encoder, 'Payment_of_Min_AMount', input_data['Payment_of_Min_Amount'])
input_data['Payment_Behaviour'] = safe_transform(payment_behaviour_encoder, 'Payment_Behaviour', input_data['Payment_Behaviour'])

# # ==== One-hot encode and align with training columns ====
# input_data = pd.get_dummies(input_data)

# Ensure input_data has the same columns and order as during training
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0  # add missing columns with default 0

input_data = input_data[model_columns]  # reorder columns

# ==== Scale input features ====
input_scaled = scaler.transform(input_data)


# ==== Predict ====
if st.button("Predict Credit Score"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Credit Score: **{prediction[0]}**")
    if prediction == 0:
        system_message = "Poor Loan Status: Do Not Grant Loan Access"
        st.error(system_message)
    elif prediction == 1:
        system_message = "Good Loan Status: Grant Loan But Consider the Other Factors with Caution"
        st.success(system_message)
    else:
        system_message = "Standard Loan Status: Grant Loan"
        st.success(system_message)

