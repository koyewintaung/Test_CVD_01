# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_model():
    return joblib.load("model.joblib")

model = load_model()
st.title("Heart-disease Explorer")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    if st.button("Predict"):
        preds = model.predict_proba(df)[:,1]
        df['pred_prob'] = preds
        st.dataframe(df[['pred_prob']].head())
