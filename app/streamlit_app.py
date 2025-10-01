import streamlit as st
import pandas as pd
import joblib

st.title("❤️ Cardiovascular Disease Predictor")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# File uploader
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data Preview:", df.head())

    if st.button("Predict"):
        preds = model.predict(df.drop(columns=["patientid"], errors="ignore"))
        df["prediction"] = preds
        st.write("Prediction Results:", df.head())
