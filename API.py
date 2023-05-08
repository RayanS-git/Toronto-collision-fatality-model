import pandas as pd
import pickle
from flask import Flask, jsonify, request
import os
import joblib

# Load the pickle model
model = joblib.load('best_model_decision.pkl')

app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # noinspection PyTypeChecker
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(request.files['file'])

    print(type(data))

    # Make predictions using the model
    predictions = model.predict(data)

    # Convert predictions to a list
    results = predictions.tolist()

    # Return the predictions as a JSON object
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

