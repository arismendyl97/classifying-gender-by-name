from prefect import flow, task
import mlflow.keras
from flask import Flask, request, jsonify
import prometheus_client
from prometheus_client import Counter
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import threading

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set the MLFlow tracking URI to a relative path
mlflow.set_tracking_uri("file:./mlruns")

@task
def load_model(model_name: str, model_version: int):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri=model_uri)
    return model

def preprocess_name(name, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([name])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence

def create_flask_app(model, tokenizer, max_sequence_length):
    app = Flask(__name__)

    # Create a counter to track the number of predictions
    prediction_counter = Counter('model_predictions_total', 'Total number of model predictions')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        name = data.get('name')
        if not name:
            return jsonify({'error': 'No name provided'}), 400
        
        # Process the name to fit the model's input requirements
        processed_name = preprocess_name(name, tokenizer, max_sequence_length)

        predictions = model.predict(processed_name)
        prediction_counter.inc()  # Increment the prediction counter
        
        predictionsList = predictions.tolist()[0]
        predictions = ["Female" if item > 0.5 else "Male" for item in predictionsList]

        # Return the prediction result
        return jsonify({'predictions': predictions})

    # @app.route('/metrics')
    # def metrics():
    #     return prometheus_client.generate_latest(), 200
    
    @app.route('/drift_report')
    def drift_report():
        return data_drift_report.json()

    return app

def run_flask_app(model, tokenizer, max_sequence_length):
    app = create_flask_app(model, tokenizer, max_sequence_length)
    app.run(host='0.0.0.0', port=1500)

@task
def start_flask_app(model, tokenizer, max_sequence_length):
    thread = threading.Thread(target=run_flask_app, args=(model, tokenizer, max_sequence_length))
    thread.start()
    return thread

@flow(name="model-serving")
def model_serving_flow(model_name: str, model_version: int, tokenizer, max_sequence_length):
    model = load_model(model_name, model_version)
    start_flask_app(model, tokenizer, max_sequence_length)

if __name__ == "__main__":

    # Initialize Evidently report for Data Drift
    data_drift_report = Report(metrics=[DataDriftPreset()])

    model_name = "GenderClassificationModel"
    model_version = 1

    with open('./training/tokenizer_info.pickle', 'rb') as handle:
        tokenizer_info = pickle.load(handle)

    tokenizer = tokenizer_info['tokenizer']
    max_sequence_length = tokenizer_info['max_sequence_length']

    model_serving_flow(model_name, model_version, tokenizer, max_sequence_length)