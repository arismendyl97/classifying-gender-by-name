import csv
import time
import requests

# Define the path to the CSV file
csv_file_path = './data_cleaning/spanish names db - post_testing.csv'  # Replace with the actual path

# API URL
api_url = "http://localhost:4500/predict"

# Open the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Loop through each row in the CSV file
    for row in reader:
        if row:  # Check if the row is not empty
            name = row[0]  # Assuming the name is in the first column
            
            # Prepare the payload for the POST request
            payload = {"name": name}
            
            # Make the POST request
            response = requests.post(api_url, json=payload)
            
            # Print the response from the server (optional)
            print(response.json())  # Or response.text if the response is not JSON
            
            # Wait for 1 second before processing the next row
            time.sleep(1)