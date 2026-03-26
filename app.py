import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Merchant Risk Analyzer", layout="wide")

st.title("💳 AI Merchant Risk Dashboard")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("merchant_risk_dataset_100k.csv")

# ---------------- LOAD MODELS ----------------
model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Merchant Details")

txn = st.sidebar.number_input("Transaction Count", min_value=1)
volume = st.sidebar.number_input("Total Volume", min_value=1.0)
refund = st.sidebar.number_input("Refund Rate", min_value=0.0, max_value=1.0)
chargeback = st.sidebar.number_input("Chargeback Rate", min_value=0.0, max_value=1.0)

category = st.sidebar.selectbox(
    "Category",
    ["E-commerce", "Gaming", "Travel", "Crypto", "Services", "Retail"]
)

desc = st.sidebar.text_input("Description")

# ---------------- NLP ----------------
def simple_nlp(text):
    text = text.lower()
    if any(word in text for word in ["crypto", "gaming", "betting"]):
        return "High"
    elif any(word in text for word in ["travel", "booking"]):
        return "Medium"
    else:
        return "Low"

# ---------------- PREDICTION ----------------
if st.sidebar.button("Analyze Merchant"):

    avg_txn = volume / txn if txn != 0 else 0
    growth = 0  # single input case

    # Create input dataframe
    X_new = pd.DataFrame([{
        "TransactionCount": txn,
        "TotalVolume": volume,
        "RefundRate": refund,
        "ChargebackRate": chargeback,
        "AvgTransactionValue": avg_txn,
        "TxnGrowthRate": growth
    }])

    # Apply SAME feature weighting (IMPORTANT)
    X_new["RefundRate"] *= 2
    X_new["ChargebackRate"] *= 2

    # Scale
    X_scaled = scaler.transform(X_new)

    # Convert to dataframe for readability
    X_scaled_df = pd.DataFrame(X_scaled, columns=[
        "TransactionCount",
        "TotalVolume",
        "RefundRate",
        "ChargebackRate",
        "AvgTransactionValue",
        "TxnGrowthRate"
    ])

    # Predict anomaly
    pred = model.predict(X_scaled)[0]
    anomaly = 1 if pred == -1 else 0

    # ---------------- RISK SCORE (FIXED) ----------------
    # Normalize manually (SAFE + INTERPRETABLE)
    avg_txn = volume / txn if txn != 0 else 0

    avg_txn_norm = avg_txn / 1000  # adjust scale
    txn_norm = txn / 5000
    volume_norm = volume / 500000

    risk_score = (
            0.35 * refund +
            0.35 * chargeback +
            0.15 * avg_txn_norm +
            0.05 * txn_norm +
            0.10 * anomaly
    )

    # clip
    risk_score = max(0, min(1, risk_score))

    # ---------------- FINAL LABEL ----------------
    if risk_score > 0.6:
        risk_label = "High"
    elif risk_score > 0.3:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    # ---------------- NLP ----------------
    nlp_risk = simple_nlp(desc)

    # ---------------- REASONS ----------------
    reasons = []

    if refund > 0.15:
        reasons.append("High refund rate")
    if chargeback > 0.1:
        reasons.append("High chargeback rate")
    if anomaly == 1:
        reasons.append("Anomalous behavior detected")
    if nlp_risk == "High":
        reasons.append("High-risk category")

    reason_text = " + ".join(reasons) if reasons else "Normal behavior"

    # ---------------- OUTPUT ----------------
    st.subheader("🔍 Risk Analysis Result")

    col1, col2 = st.columns(2)

    col1.metric("Risk Score", f"{risk_score:.2f} / 1.00")
    col2.metric("Risk Level", risk_label)

    # Progress bar (🔥 UI improvement)
    st.progress(risk_score)

    st.write(f"**Reason:** {reason_text}")

# ---------------- DASHBOARD ----------------
st.subheader("📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Merchants", len(data))

if "Final_Risk" in data.columns:
    col2.metric("High Risk", (data["Final_Risk"] == "High").sum())
else:
    col2.metric("High Risk", "N/A")

if "Anomaly_Label" in data.columns:
    col3.metric("Anomalies", (data["Anomaly_Label"] == 1).sum())
else:
    col3.metric("Anomalies", "N/A")

# ---------------- CHARTS ----------------
if "Final_Risk" in data.columns:
    st.subheader("📈 Risk Distribution")
    st.bar_chart(data["Final_Risk"].value_counts())

if "Anomaly_Label" in data.columns:
    st.subheader("⚠️ Anomaly Distribution")
    st.bar_chart(data["Anomaly_Label"].value_counts())

# ---------------- TABLE ----------------
if "Final_Risk" in data.columns:
    st.subheader("🚨 High Risk Merchants")
    st.dataframe(data[data["Final_Risk"] == "High"].head(10))