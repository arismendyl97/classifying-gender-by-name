# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

import mlflow
import mlflow.keras


import pickle

# Set the MLFlow tracking URI to a relative path
mlflow.set_tracking_uri("file:../mlruns")

# %%
def create_model(embedding_dim=128, lstm_units=64, dropout_rate=0.5):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# %%
seed = 42

# Load dataset
trainingDf = pd.read_csv('../data_cleaning/spanish names db - training.csv')
validationDf = pd.read_csv('../data_cleaning/spanish names db - validation.csv')
testingDf = pd.read_csv('../data_cleaning/spanish names db - testing.csv')

# %%
X_train = trainingDf['name']
y_train = trainingDf['gender']

X_val = validationDf['name']
y_val = validationDf['gender']

X_test = testingDf['name']
y_test  = testingDf['gender']

# %%
# Tokenize names using only the training data
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(X_train)

# Convert names to sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)

# Determine the maximum sequence length from the training data
max_sequence_length = max(len(seq) for seq in train_sequences)

# Pad sequences to the same length
X_train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
X_val_padded = pad_sequences(val_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Save tokenizer
tokenizer_info = {
    'tokenizer': tokenizer,
    'max_sequence_length': max_sequence_length
}

with open('tokenizer_info.pickle', 'wb') as handle:
    pickle.dump(tokenizer_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
from sklearn.model_selection import ParameterSampler

param_distributions = {
    'embedding_dim': [64, 128, 256],
    'lstm_units': [32, 64, 128],
    'dropout_rate': [0.3, 0.5, 0.7]
}

n_iter = 5
param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=seed))

for params in param_list:
    with mlflow.start_run():
        model = create_model(**params)
        model.fit(X_train_padded, y_train, validation_data=(X_val_padded, y_val), epochs=10, batch_size=32)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_padded, y_test)
        mlflow.log_metric('test_loss', loss)
        mlflow.log_metric('test_accuracy', accuracy)

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.keras.log_model(model, "model")