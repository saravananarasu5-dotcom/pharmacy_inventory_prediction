import streamlit as st
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
medicine_price = joblib.load("medicine_price.pkl")

st.set_page_config(page_title="Pharmacy Inventory Prediction", layout="wide")

st.title("💊 Pharmacy Inventory Prediction System")
st.write("Predict future medicine demand using Machine Learning")

# Sidebar
st.sidebar.header("User Input")

medicine_list = list(medicine_price.keys())

medicine = st.sidebar.selectbox("Select Medicine", medicine_list)

days = st.sidebar.slider("Number of Future Days", 1, 30, 7)

# Encode medicine
encoded = le.transform([medicine])[0]

price = medicine_price[medicine]

input_data = pd.DataFrame({
    "medicine_encoded": [encoded],
    "price": [price]
})

# Prediction
daily_demand = model.predict(input_data)[0]
total_demand = int(daily_demand * days)

# Display Results
st.subheader("Prediction Result")

col1, col2, col3 = st.columns(3)

col1.metric("Medicine", medicine)
col2.metric("Daily Demand", int(daily_demand))
col3.metric("Total Demand", total_demand)

st.success(f"Predicted demand for {days} days: {total_demand} units")

# Price information
st.subheader("Medicine Price")
st.write(f"Price of *{medicine}*: ₹{price}")

# Simple demand chart
data = pd.DataFrame({
    "Days": list(range(1, days+1)),
    "Predicted Demand": [int(daily_demand)] * days
})

st.subheader("Demand Forecast Chart")
st.line_chart(data.set_index("Days"))

st.markdown("---")
st.caption("AI Powered Pharmacy Inventory Prediction System")