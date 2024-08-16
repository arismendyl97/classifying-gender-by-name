import logging
from prefect import flow, task
import mlflow.keras
from flask import Flask, request, jsonify
import prometheus_client
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import threading
import pandas as pd
import subprocess  # Import subprocess module for running the additional script

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

@task
def load_model(model_name: str, model_version: int):

    # Set the MLFlow tracking URI to a relative path
    mlflow.set_tracking_uri("file:../mlruns")
    logger.info(f"Current Tracking URI: {mlflow.get_tracking_uri()}")
    
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri=model_uri)
    return model

def preprocess_name(name, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([name])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence

def create_flask_app(model, tokenizer, max_sequence_length):
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        global current_data
        data = request.json
        name = data.get('name')
        if not name:
            return jsonify({'error': 'No name provided'}), 400
        
        # Process the name to fit the model's input requirements
        processed_name = preprocess_name(name, tokenizer, max_sequence_length)

        predictions = model.predict(processed_name)
        
        predictionsList = predictions.tolist()[0]
        predictions = ["Female" if item > 0.5 else "Male" for item in predictionsList]

        newRow = pd.DataFrame([{'name': name, 'predicted_gender': int(round(predictionsList[0]))}])

        # Append the new row to the DataFrame using pd.concat
        current_data = pd.concat([current_data, newRow], ignore_index=True)

        # Return the prediction result
        return jsonify({'predictions': predictions})

    @app.route('/drift_report')
    def drift_report():
        # Here you'd pass your reference and current data to Evidently
        data_drift_report.run(reference_data=reference_data, current_data=current_data)
        drift_data = data_drift_report.as_dict()
        # Expose the drift data in a format that Prometheus can scrape
        return jsonify(drift_data)

    return app

def run_flask_app(model, tokenizer, max_sequence_length):
    app = create_flask_app(model, tokenizer, max_sequence_length)
    app.run(host='0.0.0.0', port=4500)

@task
def start_flask_app(model, tokenizer, max_sequence_length):
    thread = threading.Thread(target=run_flask_app, args=(model, tokenizer, max_sequence_length))
    thread.start()
    return thread

@flow(name="model-serving")
def model_serving_flow(model_name: str, model_version: int, tokenizer, max_sequence_length):
    model = load_model(model_name, model_version)
    start_flask_app(model, tokenizer, max_sequence_length)

    # Run the test script located at ./testing/test_prd_model.py
    subprocess.run(['python', './testing/test_prd_model.py'], check=True)

if __name__ == "__main__":

    # Initialize Evidently report for Data Drift
    data_drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    reference_data = pd.read_csv("./testing/spanish names db & predictions.csv")
    reference_data = reference_data[['name','predicted_gender']]

    # Create an empty DataFrame with specified columns and data types
    current_data = pd.DataFrame(columns=['name', 'predicted_gender'])
    # Set the data types for the columns
    current_data = current_data.astype({'name': 'string', 'predicted_gender': 'int'})

    model_name = "GenderClassificationModel"
    model_version = 1

    with open('./training/tokenizer_info.pickle', 'rb') as handle:
        tokenizer_info = pickle.load(handle)

    tokenizer = tokenizer_info['tokenizer']
    max_sequence_length = tokenizer_info['max_sequence_length']

    model_serving_flow(model_name, model_version, tokenizer, max_sequence_length)