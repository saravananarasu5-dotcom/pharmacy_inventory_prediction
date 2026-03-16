import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
medicine_price = joblib.load("medicine_price.pkl")

print("\nPharmacy Inventory Prediction System\n")

# User input
medicine = input("Enter medicine name: ")
days = int(input("Enter number of future days: "))

# Validate medicine name
if medicine not in medicine_price:
    print("Medicine not found. Please enter a valid medicine name.")
    exit()

# Get price of medicine
price = medicine_price[medicine]

# Encode medicine name
encoded_medicine = le.transform([medicine])[0]

# Create DataFrame with correct feature names
input_data = pd.DataFrame({
    "medicine_encoded": [encoded_medicine],
    "price": [price]
})

# Predict demand
daily_demand = model.predict(input_data)[0]

# Total demand for given days
total_demand = daily_demand * days

print("\nPredicted daily demand:", int(daily_demand), "units")
print("Predicted demand for", days, "days:", int(total_demand), "units")