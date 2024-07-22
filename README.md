# Name-Based Gender Classification Project

This project aims to classify the gender of individuals based on their Spanish names using a Long Short-Term Memory (LSTM) neural network. This model is particularly useful for filling in missing gender information in large customer databases, such as those of a mall's customer database.

## Project Structure

The project is organized into three main folders:

1. **dataset**
   - Contains six files with Spanish names and their corresponding gender. These files are used to build and validate the model.

2. **data_cleaning**
   - Contains a Python script named `prepare_dataset.ipynb` which reads the dataset and generates three files:
     - `spanish_names_db_training.csv`
     - `spanish_names_db_validation.csv`
     - `spanish_names_db_testing.csv`
   - This script ensures the dataset is clean and properly formatted for training, validation, and testing.

3. **training**
   - Contains the `training_model.ipynb` file which is responsible for training the LSTM model. The training process is managed and tracked using MLflow.
   - Once the training is complete, the best model is retrieved using the MLflow API in the `best_model.ipynb` file.

## Installation

### Using Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/arismendyl97/classifying-gender-by-name.git
   ```

2. Navigate to the project directory:
   ```bash
   cd classifying-gender-by-name
   ```

3. Create and activate the conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate mlopsproject
   ```

### Using Pip

1. Clone the repository:
   ```bash
   git clone https://github.com/arismendyl97/classifying-gender-by-name.git
   ```

2. Navigate to the project directory:
   ```bash
   cd classifying-gender-by-name
   ```

3. Install the required packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

1. **Prepare the Dataset**
   - Navigate to the `data_cleaning` folder.
   - Run the `prepare_dataset.ipynb` script to generate the training, validation, and testing datasets.

2. **Train the Model**
   - Navigate to the `training` folder.
   - Run the `training_model.ipynb` script to start training the LSTM model. The training process will be logged and monitored using MLflow.

3. **Retrieve the Best Model**
   - After training, run the `best_model.ipynb` script to retrieve and save the best-performing model using the MLflow API.

## Requirements

- Python 3.11.9
- TensorFlow
- MLflow
- Pandas
- NumPy

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to all contributors and the open-source community for their invaluable resources and support.