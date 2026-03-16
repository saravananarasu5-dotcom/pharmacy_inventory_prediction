import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('dataset.csv')

# Data preprocessing
# Encode medicine names
le = LabelEncoder()
df['medicine_encoded'] = le.fit_transform(df['medicine_name'])

# Features: encoded medicine and price
X = df[['medicine_encoded', 'price']]
y = df['quantity_sold']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Model Evaluation:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R2) Score: {r2:.2f}')

# Save the model
joblib.dump(model, 'model.pkl')

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')

# Save medicine prices for prediction
medicine_price = df.groupby('medicine_name')['price'].first().to_dict()
joblib.dump(medicine_price, 'medicine_price.pkl')

print('Model and encoders saved successfully.')