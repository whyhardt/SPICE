import sys, os

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


results_dir = 'benchmarking/results/'
results_file = 'results_sugawara'

dict_df = {}
# Iterate over all files in the directory
for filename in os.listdir(results_dir):
    if results_file in filename:
        file_path = os.path.join(results_dir, filename)  # Full path to the file
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            dict_df[filename] = pd.read_csv(file_path)