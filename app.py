import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 Fraud Detection App")

st.write("Enter transaction details:")

# Example: simplified inputs
amount = st.number_input("Transaction Amount")

# Dummy inputs for other features (28 V columns)
features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

if st.button("Predict"):
    input_data = np.array([[amount] + features])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")
