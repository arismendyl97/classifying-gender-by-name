FROM python:3.11-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY /app /app

# Copy the mlruns folder
COPY /mlruns ../mlruns

# Copy the requirement, data_cleaning and training folder
COPY /training /app/training
COPY /data_cleaning /app/data_cleaning

COPY requirements.txt .

# Install the necessary packages
RUN pip install -r requirements.txt

# Command to run the Prefect flow
CMD ["python","flow.py"]