# %%
from mlflow.tracking import MlflowClient
import mlflow.keras
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# Set the experiment name (or ID)
experiment_name = "Default"  # Replace with your experiment name

# Initialize MLflow client
client = MlflowClient()

# Get the experiment
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Get all runs in the experiment
runs = client.search_runs(experiment_ids=experiment_id)

# Find the best run based on test accuracy
best_run = None
best_accuracy = -float("inf")

for run in runs:
    metrics = run.data.metrics
    if 'test_accuracy' in metrics and metrics['test_accuracy'] > best_accuracy:
        best_accuracy = metrics['test_accuracy']
        best_run = run

if best_run:
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best test accuracy: {best_accuracy}")
    print("Best parameters:")
    for param in best_run.data.params:
        print(f"  {param}: {best_run.data.params[param]}")
else:
    print("No runs found.")

# %%
#Register the best model
model_uri = f"runs:/{best_run.info.run_id}/model"
model_name = "GenderClassificationModel"

registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Model registered with name: {model_name} and version: {registered_model.version}")

# %%
# Load the best model
model_version = 1

# Load the registered model
best_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{model_version}")

with open('tokenizer_info.pickle', 'rb') as handle:
    tokenizer_info = pickle.load(handle)

tokenizer = tokenizer_info['tokenizer']
max_sequence_length = tokenizer_info['max_sequence_length']

# %%
# Function to preprocess input name
def preprocess_name(name, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([name])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence

# Example name to predict
name_to_predict = "Federico"
preprocessed_name = preprocess_name(name_to_predict, tokenizer, max_sequence_length)

# Make a prediction
prediction = best_model.predict(preprocessed_name)

# Interpret the prediction
gender = "Female" if prediction[0] > 0.5 else "Male"
print(f"The predicted gender for the name '{name_to_predict}' is {gender}.")


