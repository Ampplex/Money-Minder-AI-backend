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

        # Process the data
        result = process_data(data['data'], data["category"], data["time"])
        print(result)

        # Return a JSON response
        return jsonify({'result': result})

def process_data(data, category, time):
    print(data)  # Verify the structure of the input data

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
    time_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('time', time_transformer, ['time']),
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
    input_time = datetime.strptime(time, "%Y-%m-%d")
    input_category = category

    # Convert input time to UNIX timestamp
    input_time_unix = input_time.timestamp() // 10**9

    # Transform input data using the preprocessor
    input_data = pd.DataFrame({'time': [input_time_unix]})
    predicted_amount = pipeline.predict(input_data)

    print(f"Predicted Expense Amount for {input_category} on {input_time} : {predicted_amount}")
    return predicted_amount[0]


if __name__ == '__main__':
    app.run(debug=True)
