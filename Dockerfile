FROM python:3.11-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the necessary packages
RUN pip install -r requirements.txt

# Copy the Prefect flow script
COPY flow.py /app/flow.py

# Command to run the Prefect flow
CMD ["python", "flow.py"]