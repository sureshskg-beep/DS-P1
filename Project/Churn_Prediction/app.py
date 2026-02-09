import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn")

tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Placeholder for remaining encoded features
other_features = np.zeros(16)

if st.button("Predict"):
    input_data = np.hstack([[tenure, monthly_charges, total_charges], other_features])
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")
