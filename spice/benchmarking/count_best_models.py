import pandas as pd
import os

# Directory containing the CSV files
csv_directory = "benchmarking/results"
base_name = "results_sugawara"

# Dictionary to hold all data across files
participant_data = {}

# Iterate through all CSV files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv") and '_rnn' not in filename and base_name in filename:
        filepath = os.path.join(csv_directory, filename)
        model_name = os.path.splitext(filename)[0]  # Use filename (without extension) as model name

        # Read the CSV file
        data = pd.read_csv(filepath)

        # Add NLL for each participant under the corresponding model
        for participant_id, nll in data[['Job_ID', 'AIC']].values:
            if participant_id not in participant_data:
                participant_data[participant_id] = {}
            participant_data[participant_id][model_name] = nll

# Dictionary to count the number of times each model has the lowest NLL
model_lowest_nll_count = {}

# Compare NLL across models for each participant
for participant_id, models in participant_data.items():
    # Find the model with the lowest NLL for the participant
    lowest_model = min(models, key=models.get)
    model_lowest_nll_count[lowest_model] = model_lowest_nll_count.get(lowest_model, 0) + 1

# Print results
for model, count in model_lowest_nll_count.items():
    print(f"Model {model} had the lowest NLL for {count} participants.")
