import csv
import time
import os

# Define the path to the CSV file
csv_file_path = './data_cleaning/spanish names db - post_testing.csv'  # Replace with the actual path

# API URL and curl command template
api_url = "http://localhost:4500/predict"
curl_command_template = 'curl -X POST {url} -H "Content-Type: application/json" -d \'{{"name": "{name}"}}\''

# Open the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Loop through each row in the CSV file
    for row in reader:
        if row:  # Check if the row is not empty
            name = row[0]  # Assuming the name is in the first column
            
            # Prepare the curl command
            curl_command = curl_command_template.format(url=api_url, name=name)
            
            # Execute the curl command
            os.system(curl_command)
            
            # Wait for 1 second before processing the next row
            time.sleep(1)