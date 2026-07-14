import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Saved Files
# -----------------------------
model = joblib.load("model_predictor.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
body_encoder = joblib.load("body_encoder.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Car Model Predictor", page_icon="🚗")

st.title("🚗 Car Model Predictor")
st.write("Predict the car model using Logistic Regression.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------

body_type = st.selectbox(
    "Body Type",
    body_encoder.classes_
)

engine_cc = st.number_input(
    "Engine Capacity (cc)",
    min_value=500,
    max_value=8000,
    value=1500
)

power_bhp = st.number_input(
    "Power (BHP)",
    min_value=20,
    max_value=1200,
    value=100
)

mileage_kmpl = st.number_input(
    "Mileage (km/l)",
    min_value=5.0,
    max_value=40.0,
    value=18.0
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Car Model"):

    # Encode body type
    body_encoded = body_encoder.transform([body_type])[0]

    # Create feature array
    features = np.array([[
        body_encoded,
        engine_cc,
        power_bhp,
        mileage_kmpl
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)

    # Convert prediction back to model name
    car_model = label_encoder.inverse_transform(prediction)[0]

    # Confidence
    probability = model.predict_proba(features_scaled)
    confidence = np.max(probability) * 100

    st.success(f"### 🚗 Predicted Car Model: {car_model}")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )