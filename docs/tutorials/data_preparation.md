---
layout: default
title: Data Preparation in SPICE
parent: Tutorials
nav_order: 1
---

# Data Preparation in SPICE

This tutorial covers how to prepare and manipulate datasets for use with SPICE. We'll explore the core data structures and utilities that SPICE provides for handling experimental data.

## Prerequisites

- Basic understanding of Python and PyTorch
- SPICE installed in your environment
- Basic familiarity with cognitive science experimental data

## The DatasetRNN Class

The `DatasetRNN` class is the fundamental data structure in SPICE for handling experimental data. It inherits from PyTorch's `Dataset` class and is specifically designed for working with sequential data.

```python
from spice.resources.rnn_utils import DatasetRNN
import torch

# Example of creating a DatasetRNN instance
xs = torch.zeros((n_sessions, n_timesteps, n_features))  # Input features
ys = torch.zeros((n_sessions, n_timesteps, n_actions))   # Target actions
dataset = DatasetRNN(
    xs=xs,
    ys=ys,
    normalize_features=(0, 1),  # Optional: Normalize specific feature dimensions
    sequence_length=50,         # Optional: Set fixed sequence length
    stride=1,                   # Optional: Stride for sequence creation
    device='cuda'               # Optional: Specify torch device
)
```

### Key Features of DatasetRNN

1. **Shape Requirements**:
   - `xs`: Shape (n_sessions, n_timesteps, n_features)
   - `ys`: Shape (n_sessions, n_timesteps, n_actions)

2. **Automatic Feature Normalization**:
   - Specify which feature dimensions to normalize using `normalize_features`
   - Normalization is performed per feature to range [0, 1]

3. **Sequence Processing**:
   - Optional fixed sequence length with `sequence_length`
   - Configurable stride for sequence creation
   - Automatic handling of variable-length sequences

## Converting Experimental Data

SPICE provides a utility function `convert_dataset()` to transform experimental data given in a CSV file into the `DatasetRNN` format. This function handles various aspects of data preprocessing and normalization:

```python
from spice.utils.convert_dataset import convert_dataset

# Convert a CSV file containing experimental data
dataset, experiment_list, df, dynamics = convert_dataset(
    file="experiment_data.csv",           # Path to CSV file
    device=None,                          # Optional: PyTorch device
    sequence_length=None,                 # Optional: Fixed sequence length
    df_participant_id='session',          # Column name for participant/session ID
    df_block='block',                     # Column name for block information
    df_experiment_id='experiment',        # Column name for experiment ID
    df_choice='choice',                   # Column name for choices
    df_reward='reward',                   # Column name for rewards
    additional_inputs=['context']         # Optional: Additional input columns
)
```

### What convert_dataset() Does

1. **Data Loading and Preprocessing**:
   - Loads the CSV file into a pandas DataFrame
   - Replaces all NaN values with -1
   - Maintains a copy of the original DataFrame

2. **ID and Block Processing**:
   - Converts participant IDs to numeric values (0-based indexing)
   - Similarly processes experiment IDs if present
   - Normalizes block numbers to [0,1] range if present

3. **Choice Processing**:
   - Shifts choices to start from 0
   - Creates one-hot encoded representations

4. **Reward Processing**:
   - Normalizes rewards to [0,1] range
   - Handles both standard and counterfactual rewards
   - Supports multiple reward columns (e.g., 'reward_0', 'reward_1', etc.)

5. **Additional Features**:
   - Processes any specified additional input columns
   - Maintains proper alignment of all features

### Return Values

The function returns a tuple of four elements:

1. `dataset`: A DatasetRNN instance containing:
   - Input features (xs): Choices, rewards, blocks, IDs, and additional inputs
   - Target actions (ys): One-hot encoded next choices

2. `experiment_list`: List of BanditSession objects containing:
   - Raw choices and rewards
   - Session information
   - Trial counts

3. `df`: Original DataFrame preserved for reference

4. `dynamics`: Tuple of arrays containing behavioral metrics:
   - Choice probabilities
   - Action values
   - Reward values
   - Choice values
   (Note: These are only populated for datasets generated with specific tools)

### Example Usage with Additional Features

```python
# Example with multiple features and counterfactual rewards
dataset, experiments, df, dynamics = convert_dataset(
    file="experiment_data.csv",
    device='cuda',
    df_participant_id='subject_id',
    df_block='block_num',
    df_experiment_id='condition',
    df_choice='chosen_option',
    df_reward='obtained_reward',
    additional_inputs=['context', 'stimulus']
)

# Check the shape of processed data
print(f"Input features shape: {dataset.xs.shape}")
print(f"Target actions shape: {dataset.ys.shape}")
print(f"Number of experiments: {len(experiments)}")
```

## Splitting Datasets

SPICE provides two main methods for splitting datasets for training and testing:

### 1. Splitting Along Time Dimension

This method splits each session's data into training and testing sets based on time:

```python
from spice.resources.rnn_utils import split_data_along_timedim

# Split dataset with 80% of timesteps for training
train_dataset, test_dataset = split_data_along_timedim(
    dataset=dataset,
    split_ratio=0.8,
    device=torch.device('cpu')
)
```

This approach:
- Preserves session structure
- Splits each session at the same relative timepoint
- Handles variable-length sessions appropriately
- Maintains padding (-1) for shorter sessions

### 2. Splitting Along Session Dimension

For splitting based on participants/sessions:

```python
# Assuming dataset.xs and dataset.ys contain your data
n_sessions = dataset.xs.shape[0]
split_idx = int(0.8 * n_sessions)

# Create training dataset
train_dataset = DatasetRNN(
    xs=dataset.xs[:split_idx],
    ys=dataset.ys[:split_idx],
    device=dataset.device
)

# Create test dataset
test_dataset = DatasetRNN(
    xs=dataset.xs[split_idx:],
    ys=dataset.ys[split_idx:],
    device=dataset.device
)
```

This approach:
- Keeps entire sessions together
- Useful for testing generalization across participants
- Maintains all temporal information within sessions

## Best Practices

1. **Data Preprocessing**:
   - Always check for missing values in your raw data
   - Ensure consistent data types across sessions
   - Normalize features when appropriate
   - Use sequence_length parameter for long sequences

2. **Splitting Strategy**:
   - Use time-based splits when testing temporal generalization
   - Use session-based splits when testing participant generalization
   - Consider your research question when choosing split method

## Example: Complete Data Pipeline

Here's a complete example showing the entire data preparation pipeline:

```python
import torch
from spice.utils.convert_dataset import convert_dataset
from spice.resources.rnn_utils import split_data_along_timedim

# 1. Load and convert data
dataset, experiments, df, dynamics = convert_dataset(
    file="experiment_data.csv",
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    sequence_length=100,  # Set fixed sequence length
    df_participant_id='participant',
    df_choice='choice',
    df_reward='reward'
)

# 2. Split data for training and testing
train_dataset, test_dataset = split_data_along_timedim(
    dataset=dataset,
    split_ratio=0.8,
    device=dataset.device
)

# 3. Verify data shapes
print(f"Training data shape: {train_dataset.xs.shape}")
print(f"Testing data shape: {test_dataset.xs.shape}")

# 4. Create DataLoader for batch processing
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

## Next Steps

- Learn how to [train a basic Rescorla-Wagner model](rescorla_wagner.html)
- Understand how to [incorporate custom mechanisms such as forgetting](rescorla_wagner_forgetting.html)
- Explore [working with hardcoded equations](hardcoded_equations.html)