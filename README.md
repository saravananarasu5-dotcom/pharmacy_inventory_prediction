# Pharmacy Inventory Prediction System

## Project Objective
This project aims to predict future medicine demand in a pharmacy using historical sales data and machine learning techniques. It helps pharmacy managers optimize inventory levels and reduce stockouts or overstocking.

## Technologies Used
- **Python**: Programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library (Linear Regression model)
- **matplotlib**: Data visualization
- **joblib**: Model serialization

## Machine Learning Model
The system uses a Linear Regression model to predict daily medicine demand based on:
- Encoded medicine name
- Medicine price

The model is trained on historical sales data and evaluated using:
- Mean Absolute Error (MAE)
- R-squared (R2) Score

## How to Run the Project

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Train the Model** (if not already trained):
   ```
   python train_model.py
   ```

3. **Run Predictions**:
   ```
   python predict_inventory.py
   ```
   Enter the medicine name and number of future days when prompted.

4. **View Visualizations**:
   ```
   python visualization.py
   ```

## Project Structure
```
pharmacy_inventory_prediction/
├── dataset.csv                 # Synthetic pharmacy sales data
├── data_preprocessing.py       # (Integrated in train_model.py)
├── train_model.py              # Model training and evaluation
├── predict_inventory.py        # Console prediction program
├── visualization.py            # Data visualization scripts
├── model.pkl                   # Trained model file
├── label_encoder.pkl           # Label encoder for medicine names
├── medicine_price.pkl          # Medicine price dictionary
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Example Output
```
Enter medicine name: Paracetamol
Enter number of future days: 7

Predicted demand: 120 units
```

## Notes
- The dataset contains synthetic data with 200 rows of pharmacy sales.
- The model predicts daily demand multiplied by the number of days.
- Visualizations include historical sales trends and medicine-wise sales totals.