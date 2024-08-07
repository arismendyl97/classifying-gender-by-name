from prefect import flow, task
import mlflow.keras
from flask import Flask, request, jsonify
import prometheus_client
from prometheus_client import Counter
import threading

# Set the MLFlow tracking URI to a relative path
mlflow.set_tracking_uri("file:./mlruns")

@task
def load_model(model_name: str, model_version: int):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri=model_uri)
    return model

def create_flask_app(model):
    app = Flask(__name__)

    # Create a counter to track the number of predictions
    prediction_counter = Counter('model_predictions_total', 'Total number of model predictions')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        predictions = model.predict(data['instances'])
        prediction_counter.inc()  # Increment the prediction counter
        return jsonify(predictions.tolist())

    @app.route('/metrics')
    def metrics():
        return prometheus_client.generate_latest(), 200

    return app

def run_flask_app(model):
    app = create_flask_app(model)
    app.run(host='0.0.0.0', port=1500)

@task
def start_flask_app(model):
    thread = threading.Thread(target=run_flask_app, args=(model,))
    thread.start()
    return thread

@flow(name="model-serving")
def model_serving_flow(model_name: str, model_version: int):
    model = load_model(model_name, model_version)
    start_flask_app(model)

if __name__ == "__main__":
    model_name = "GenderClassificationModel"
    model_version = 1
    model_serving_flow(model_name, model_version)