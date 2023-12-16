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
    {"amount": 5000, "time": "2023-12-15T18:08:21.258+00:00", "category": "Groceries"},
    {"amount": 10, "time": "2023-12-15T18:07:14.119+00:00", "category": "Clothing"},
    {"amount": 300, "time": "2023-12-14T09:52:23.883+00:00", "category": "Groceries"},
    {"amount": 5000, "time": "2023-12-14T09:52:11.664+00:00", "category": "Clothing"},
    {"amount": 100, "time": "2022-03-14T09:52:11.664+00:00", "category": "Groceries"},
    {"amount": 4000, "time": "2023-08-14T09:52:11.664+00:00", "category": "Groceries"},
    {"amount": 500, "time": "2023-02-14T09:52:11.664+00:00", "category": "Groceries"},
    {"amount": 5000, "time": "2023-01-14T09:52:11.664+00:00", "category": "Groceries"},
]

# Convert data to DataFrame
df = pd.DataFrame(data)
df["time"] = pd.to_datetime(df["time"])

# Convert Timestamps to Unix timestamps (int64)
df["time"] = (df["time"].astype(np.int64) // 10**9)

# Encode the "Category" variable
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["category"])

# Extract features (time and category) and target variable (amount)
X = df[["time", "category_encoded"]]
y = df["amount"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a ColumnTransformer with a StandardScaler for the time feature
preprocessor = ColumnTransformer(
    transformers=[
        ('time', 'passthrough', ['time']),
        ('category', 'passthrough', ['category_encoded'])  # Include category in the model
    ]
)

# Create a linear regression model
model = LinearRegression()

# Create a pipeline with preprocessing and the linear regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Input data for predicting future expenses (adjust as needed)
input_time = datetime.strptime("2024-03-22", "%Y-%m-%d")
input_category = "Clothing"

# Convert input time to UNIX timestamp
input_time_unix = input_time.timestamp() // 10**9

# Encode the input category
input_category_encoded = label_encoder.transform([input_category])

# Transform input data using the preprocessor
input_data = pd.DataFrame({'time': [input_time_unix], 'category_encoded': input_category_encoded})
predicted_amount = pipeline.predict(input_data)

print(predicted_amount)
