from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['POST'])
def ExpensePredictor():
    if request.method == 'POST':
        # Access the data sent with the POST request
        data = request.json  # Assuming the data is in JSON format
        # print(data["data"],data["category"])

        # Process the data
        result = process_data(data["data"], data["category"])
        return jsonify({'result': result})


def process_data(data, category):
    print(data)  # Verify the structure of the input data

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    df["category"] = category
        

    df["time"] = pd.to_datetime(df["time"])

    # Convert Timestamps to Unix timestamps (int64)
    df["time_unix"] = (df["time"].astype(np.int64) // 10**9)

    # Encode the "Category" variable
    label_encoder = LabelEncoder()
    df["category_encoded"] = label_encoder.fit_transform(df["category"])

    # Extract features (time and category) and target variable (amount)
    X = df[["time_unix", "category_encoded"]]
    y = df["amount"]

    # Create a ColumnTransformer with a StandardScaler for the time feature
    time_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('time', time_transformer, ['time_unix']),
        ]
    )

    # Create a linear regression model
    model = LinearRegression()

    # Create a pipeline with preprocessing and the linear regression model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model using the entire dataset
    pipeline.fit(X, y)

    # Input data for predicting future expenses (adjust as needed)
    input_time = pd.to_datetime(df['time'].iloc[0])
    input_category = df['category'].iloc[0]

    # Convert input time to UNIX timestamp
    input_time_unix = input_time.timestamp() // 10**9

    # Transform input data using the preprocessor
    input_data = pd.DataFrame({'time_unix': [input_time_unix]})
    predicted_amount = pipeline.predict(input_data)

    print(f"Predicted Expense Amount for {input_category} on {input_time} : {predicted_amount}")
    return {'predicted_amount': predicted_amount[0]}


if __name__ == '__main__':
    app.run(debug=True)
