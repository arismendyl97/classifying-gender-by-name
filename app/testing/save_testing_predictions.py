import mlflow.keras
import pandas as pd
from keras.models import load_model

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_names(names, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences(names)
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence


with open('../training/tokenizer_info.pickle', 'rb') as handle:
    tokenizer_info = pickle.load(handle)

tokenizer = tokenizer_info['tokenizer']
max_sequence_length = tokenizer_info['max_sequence_length']

# Set the MLFlow tracking URI to a relative path
mlflow.set_tracking_uri("file:../mlruns")

model_name = "GenderClassificationModel"
model_version = 1

# Load the model from MLflow
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version
model = mlflow.keras.load_model(model_uri)

# Load your dataset
input_csv_path = '../data_cleaning/spanish names db - testing.csv'  # Replace with the actual path
df = pd.read_csv(input_csv_path)

# Add a column 'predicted_gender' based on the model's predictions
names = df['name'].values

preprocessed_names = preprocess_names(names, tokenizer, max_sequence_length)
# Assuming the model was trained with input data preprocessed as required
predictions = model.predict(preprocessed_names)
rounded_predictions = predictions.round().astype(int)

# Convert predictions to a readable format
# Assuming the output is a one-hot encoded or probability array
df['predicted_gender'] = rounded_predictions[:, 0]
df.rename(columns={'gender':'actual_gender'}, inplace=True)

# Save the result to a new CSV file
output_csv_path = 'spanish names db & predictions.csv'  # Replace with the desired output path
df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")