import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Plot 1: Historical daily sales
plt.figure(figsize=(12, 6))
daily_sales = df.groupby('date')['quantity_sold'].sum()
daily_sales.plot()
plt.title('Historical Daily Sales')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Total sales per medicine
plt.figure(figsize=(10, 6))
medicine_sales = df.groupby('medicine_name')['quantity_sold'].sum().sort_values(ascending=False)
medicine_sales.plot(kind='bar')
plt.title('Total Sales per Medicine')
plt.xlabel('Medicine Name')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print('Visualizations displayed.')