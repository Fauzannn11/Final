import streamlit as st
import pandas as pd
import joblib

# ======================
# Load model
# ======================
@st.cache_resource
def load_model():
    return joblib.load("customer_satisfaction_model.pkl")

model = load_model()

# ======================
# UI Header
# ======================
st.title("📊 Customer Satisfaction Prediction Dashboard")
st.caption("Dashboard prediksi kepuasan pelanggan berbasis Machine Learning")

# ======================
# Input Form
# ======================
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    total_spend = st.number_input("Total Spend", min_value=0.0, value=500.0)
    items_purchased = st.number_input("Items Purchased", min_value=1, value=5)
    avg_rating = st.number_input("Average Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    days_since = st.number_input("Days Since Last Purchase", min_value=0, value=10)

    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "Miami"])
    membership = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold"])
    discount = st.selectbox("Discount Applied", ["True", "False"])

    submit = st.form_submit_button("Predict")

# ======================
# Prediction + Dashboard
# ======================
if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "Total Spend": total_spend,
        "Items Purchased": items_purchased,
        "Average Rating": avg_rating,
        "Days Since Last Purchase": days_since,
        "Gender": gender,
        "City": city,
        "Membership Type": membership,
        "Discount Applied": discount
    }])

    pred = model.predict(input_df)[0]
    conf = model.predict_proba(input_df).max()

    # ======================
    # KPI Dashboard
    # ======================
    st.subheader("📌 Prediction Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Satisfaction", pred)
    with col2:
        st.metric("Confidence Score", f"{conf:.2f}")

    # ======================
    # Business Interpretation
    # ======================
    st.subheader("🧠 Business Insight")

    if pred == "Unsatisfied":
        st.error("⚠️ Pelanggan berisiko tidak puas. Disarankan dilakukan intervensi.")
    elif pred == "Neutral":
        st.warning("⚠️ Pelanggan berada di zona abu-abu. Perlu perhatian.")
    else:
        st.success("✅ Pelanggan puas. Fokus pada retensi dan loyalitas.")

    # ======================
    # Feature Snapshot
    # ======================
    st.subheader("📄 Input Feature Summary")
    st.table(input_df)
