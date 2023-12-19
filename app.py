from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime

app = Flask(__name__)
# testing
# Initialize an empty DataFrame for model training
model_data = pd.DataFrame(columns=["amount", "time"])

@app.route('/', methods=['POST'])
def ExpensePredictor():
    if request.method == 'POST':
        # Access the data sent with the POST request
        data = request.json  # Assuming the data is in JSON format

        # Process the data for training
        train_model(data["data"])

        # Process the data for prediction
        result = process_data(data["predict_data"])
        return jsonify({'result': result})


def train_model(train_data):
    global model_data
    # Convert training data to DataFrame
    df = pd.DataFrame(train_data)
    df["time"] = pd.to_datetime(df["time"])

    # Concatenate with existing model_data
    model_data = pd.concat([model_data, df], ignore_index=True)

    # Extract features (time) and target variable (amount)
    model_data['day_of_week'] = model_data['time'].dt.dayofweek
    X_train = model_data[["time", "day_of_week"]]
    y_train = model_data["amount"]

    # Convert Timestamps to Unix timestamps (int64)
    X_train["time"] = (X_train["time"].astype(np.int64) // 10**9)

    # Create and train the linear regression model
    global model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)


def process_data(predict_data):
    # Convert prediction data to DataFrame
    df = pd.DataFrame(predict_data)
    df["time"] = pd.to_datetime(df["time"])

    # Extract features (time) for prediction
    df['day_of_week'] = df['time'].dt.dayofweek
    X_predict = df[["time", "day_of_week"]]

    # Convert Timestamps to Unix timestamps (int64)
    X_predict["time"] = (X_predict["time"].astype(np.int64) // 10**9)

    # Predict using the trained model
    predicted_amount = model.predict(X_predict)

    print(predicted_amount)

    return {'predicted_amount': predicted_amount.tolist()}

@app.route('/server_activator')
def server_activator():
    return jsonify({'msg': 'AI activated'})

if __name__ == '__main__':
    app.run(debug=True)
