import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from datetime import datetime

# Sample data
data = [
    {"amount": 20, "time": "2023-12-15T18:08:21.258+00:00", "category": "Groceries"},
    {"amount": 10, "time": "2023-12-15T18:07:14.119+00:00", "category": "Clothing"},
    {"amount": 30, "time": "2023-12-14T09:52:23.883+00:00", "category": "Groceries"},
    {"amount": 300, "time": "2023-12-14T09:52:11.664+00:00", "category": "Clothing"},
    {"amount": 1, "time": "2022-03-14T09:52:11.664+00:00", "category": "Groceries"},
    {"amount": 40, "time": "2023-08-14T09:52:11.664+00:00", "category": "Groceries"},
    {"amount": 40, "time": "2023-02-14T09:52:11.664+00:00", "category": "Groceries"},
    {"amount": 50, "time": "2023-01-14T09:52:11.664+00:00", "category": "Groceries"},
]

# Convert data to DataFrame
df = pd.DataFrame(data)
df["time"] = pd.to_datetime(df["time"])

# Extract features (time and category) and target variable (amount)
df['day_of_week'] = df['time'].dt.dayofweek  # add day of week as a feature
X = df[["time", "category", "day_of_week"]]
y = df["amount"]

# Encode the "Category" variable
label_encoder = LabelEncoder()
X["category_encoded"] = label_encoder.fit_transform(X["category"])

# Convert Timestamps to Unix timestamps (int64)
X["time"] = (X["time"].astype(np.int64) // 10**9)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[["time", "category_encoded", "day_of_week"]], y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Input data for predicting future expenses (adjust as needed)
input_time = datetime.strptime("2024-12-20", "%Y-%m-%d")
input_category = "Groceries"
input_day_of_week = input_time.weekday()

# Convert input time to UNIX timestamp
input_time_unix = input_time.timestamp() // 10**9

# Encode the input category
input_category_encoded = label_encoder.transform([input_category])

# Predict using the trained model
predicted_amount = model.predict([[input_time_unix, input_category_encoded[0], input_day_of_week]])

print(predicted_amount)
