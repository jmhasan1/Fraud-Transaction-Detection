import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import predict_model


st.set_page_config(page_title="Fraud Transaction Detection", page_icon="💳", layout="centered")
st.title("💳 Fraud Transaction Detection")
st.markdown(
    """
    This app allows you to test the fraud detection model locally.
    You can upload a CSV of transactions or input details manually.
    """
)

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV with transactions", type=["csv"])

if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Transactions")
    st.dataframe(df_new.head())

    # Choose model
    model_name = st.selectbox("Select Model", ["XGBoost", "LightGBM", "LR", "RF_baseline", "Ensemble"])

    if st.button("Predict Fraud"):
        proba, preds = predict_model(df_new, model_name)
        df_new["fraud_proba"] = proba
        df_new["predicted_fraud"] = preds

        st.subheader("Predictions")
        st.dataframe(df_new)

        # Download CSV
        csv = df_new.to_csv(index=False).encode()
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="transactions_predicted.csv",
            mime="text/csv"
        )

# --- Manual input ---
st.subheader("Or Input a Single Transaction Manually")
with st.form("manual_input"):
    TX_AMOUNT = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    CUSTOMER_ID = st.text_input("Customer ID", value="C12345")
    TERMINAL_ID = st.text_input("Terminal ID", value="T67890")
    TX_DAY = st.number_input("Transaction Day", min_value=0, value=1)
    TX_HOUR = st.number_input("Transaction Hour", min_value=0, max_value=23, value=12)
    submitted = st.form_submit_button("Predict")

    if submitted:
        df_single = pd.DataFrame([{
            "TX_AMOUNT": TX_AMOUNT,
            "CUSTOMER_ID": CUSTOMER_ID,
            "TERMINAL_ID": TERMINAL_ID,
            "TX_DAY": TX_DAY,
            "TX_HOUR": TX_HOUR
        }])
        proba, pred = predict_model(df_single, "XGBoost")
        st.write(f"Fraud Probability: {proba[0]:.4f}")
        st.write(f"Predicted Fraud: {'Yes' if pred[0] == 1 else 'No'}")
