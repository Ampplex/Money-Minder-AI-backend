import requests
from datetime import datetime

# Replace the URL with the actual URL of your Flask server
url = 'https://money-minder-ai.onrender.com/'

# Data to be sent with the POST request (in JSON format)
data = [
    {"amount": 5000, "time": "2023-12-15T18:08:21.258+00:00", "category": "Groceries"},
    {"amount": 1000, "time": "2023-10-15T18:07:14.119+00:00", "category": "Clothing"},
    {"amount": 8000, "time": "2023-08-14T09:52:23.883+00:00", "category": "Groceries"},
    {"amount": 40, "time": "2023-12-14T09:52:11.664+00:00", "category": "Clothing"},
]

# Convert the 'time' field in postData to timestamp
postData = {
    "data": data,
    "category": "Groceries",
    "time": "2024-01-22"
}

# Send the POST request
response = requests.post(url, json=postData)

# Check the response
if response.status_code == 200:
    # Request was successful
    result = response.json()
    print('Server Response:', result)
else:
    print('Error:', response.status_code)
