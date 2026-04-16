import streamlit as st
import numpy as np
import pickle

# ==============================
# LOAD MODEL & SCALER
# ==============================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 AI Fraud Detection System")
st.markdown("Detect fraudulent transactions using Machine Learning")

# ==============================
# INPUT SECTION
# ==============================
st.header("Enter Transaction Details")

# Time + Amount (IMPORTANT FIX)
time = st.number_input("⏱️ Time", value=0.0)
amount = st.number_input("💰 Transaction Amount", value=0.0)

st.subheader("Transaction Features (V1 - V28)")

features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict Transaction"):

    try:
        # Combine inputs (FIXED: includes Time)
        input_data = np.array([[time, amount] + features])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected!\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction\n\nProbability of Fraud: {probability:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.info("This app uses an ML model trained on financial transaction data.")
