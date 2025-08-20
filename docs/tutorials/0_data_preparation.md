---
layout: default
title: Data Preparation in SPICE
parent: Tutorials
nav_order: 1
---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/whyhardt/SPICE/blob/main/tutorials/0_data_preparation.ipynb)

# Data Preparation in SPICE

This tutorial covers how to prepare and manipulate datasets for use with SPICE. We’ll explore the core data structures and utilities that SPICE provides for handling experimental data. We'll cover:

1. Preparing raw experimental data
2. Converting experimental data to SPICE format
3. Creating synthetic datasets
4. DatasetRNN class
5. Splitting data along time and session dimensions

## Prerequisites

Before starting this tutorial, make sure you have:
- SPICE and required dependencies (pandas, numpy, etc.) installed
- Basic understanding of reinforcement learning data
- Basic understanding of Python and PyTorch data structures


```python
import sys
import os
import pandas as pd
import numpy as np
from spice.resources.bandits import create_dataset, BanditsDrift, get_update_dynamics
from spice.resources.rnn_utils import DatasetRNN
from spice.utils.plotting import plot_session
```

    Library setup is valid. All keys and features appear in the provided list of features.
    Library setup is valid. All keys and features appear in the provided list of features.
    Library setup is valid. All keys and features appear in the provided list of features.
    Library setup is valid. All keys and features appear in the provided list of features.
    Library setup is valid. All keys and features appear in the provided list of features.


    /Users/imtezcan/Repositories/CogSci/SPICE/.venv/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## 1. Preparing Raw Experimental Data

SPICE expects data in a specific format for training and analysis. The basic requirements are:

- Data should be in CSV format
- Column names can be customized by setting `df_participant_id`, `df_block`, `df_experiment_id`, `df_choice` and `df_reward`.
- Additional inputs can be given as a list of strings (`additional_inputs`) corresponding to column names
- Required columns:
  - `df_participant_id (default: 'session')`: Unique identifier for each experimental session/participant
  - `df_choice (default: 'choice')`: The action taken by the participant (0-indexed)
  - `df_reward (default: 'reward')`: The reward received for the action

Let's look at an example of properly formatted data:


```python
# Create a sample dataset
sample_data = {
    'session': [1, 1, 1, 2, 2, 2],
    'choice': [0, 1, 0, 1, 0, 1],
    'reward': [1, 0, 1, 0, 1, 0],
    'rt': [0.5, 0.6, 0.4, 0.7, 0.5, 0.6]
}

df = pd.DataFrame(sample_data)
print("Sample data format:")
display(df)
```

    Sample data format:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session</th>
      <th>choice</th>
      <th>reward</th>
      <th>rt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0.6</td>
    </tr>
  </tbody>
</table>
</div>


Let's save it as a .csv file.


```python
df.to_csv('sample_data.csv', index=False)
```

## 2. Converting Experimental Data to SPICE Format

SPICE provides a utility function `convert_dataset()` to transform experimental data given in a CSV file into the `DatasetRNN` format. This function handles various aspects of data preprocessing and normalization, with high flexibility for customization. You can call the function as follows:


```python
from spice.utils.convert_dataset import convert_dataset
```


```python
dataset, experiment_list, df, dynamics = convert_dataset(
    file="sample_data.csv",           # Path to CSV file
    device=None,                          # (Optional) PyTorch device
    sequence_length=None,                 # (Optional) Fixed sequence length
    df_participant_id='session',          # (Optional) Column name for participant/session ID
    df_block='block',                     # (Optional) Column name for block information
    df_experiment_id='experiment',        # (Optional) Column name for experiment ID
    df_choice='choice',                   # (Optional) Column name for choices
    df_reward='reward',                   # (Optional) Column name for rewards
    additional_inputs=['context']         # (Optional) Additional input columns
)
```

Alternatively, if your .csv file is in the right format and adheres to defaul values, all you need to do is to provide the filename:


```python
from spice.utils.convert_dataset import convert_dataset

dataset, experiment_list, df, dynamics = convert_dataset(file='sample_data.csv')
```

The function is also highly customable, with additional parameters to specify how the data should be processed:


```python
print(dataset.xs.shape)
print(dataset.ys.shape)
```

    torch.Size([2, 3, 7])
    torch.Size([2, 3, 2])


## 3. Creating Synthetic Datasets

SPICE provides utilities to create synthetic datasets for testing and validation. Here's how to create a synthetic dataset using a simple bandit task:


```python
from spice.resources.bandits import AgentQ

# Create a simple Q-learning agent
agent = AgentQ(
    beta_reward=1.0,
    alpha_reward=0.5,
    alpha_penalty=0.5
)

# Create environment
environment = BanditsDrift(sigma=0.2)

# Generate synthetic data
n_sessions = 2
n_trials = 10

dataset, experiments, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=n_trials,
    n_sessions=n_sessions,
    verbose=False
)

# Convert to DataFrame
synthetic_data = []
for i in range(len(dataset)):
    experiment = dataset.xs[i].numpy()
    session_data = pd.DataFrame({
        'session': [i] * n_trials,
        'choice': np.argmax(experiment[:, :2], axis=1),
        'reward': np.max(experiment[:, 2:4], axis=1)
    })
    synthetic_data.append(session_data)

synthetic_df = pd.concat(synthetic_data, ignore_index=True)
print("Synthetic dataset:")
display(synthetic_df)
```

    Creating dataset...


    100%|██████████| 2/2 [00:00<00:00, 647.42it/s]

    Synthetic dataset:


    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session</th>
      <th>choice</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


Let's remove the generated file


```python
import os

os.remove('sample_data.csv')
```

## 4. The DatasetRNN Class

The `DatasetRNN` class is the fundamental data structure in SPICE for handling experimental data. It inherits from PyTorch's `Dataset` class and is specifically designed for working with sequential data. When you call `convert_dataset()` or `create_dataset()`, it returns an instance of `DatasetRNN`. To manually create a `DatasetRNN`, you can use the following constructor:


```python
from spice.resources.rnn_utils import DatasetRNN
import torch

n_sessions = 10  # Number of sessions
n_timesteps = 100  # Number of timesteps per session
n_features = 5  # Number of input features
n_actions = 3  # Number of possible actions

# Example of creating a DatasetRNN instance
xs = torch.zeros((n_sessions, n_timesteps, n_features))  # Input features
ys = torch.zeros((n_sessions, n_timesteps, n_actions))   # Target actions
dataset = DatasetRNN(
    xs=xs,
    ys=ys,
    normalize_features=(0, 1),  # (Optional) Normalize specific feature dimensions
    sequence_length=50,         # (Optional) Set fixed sequence length
    stride=1,                   # (Optional) Stride for sequence creation
    device='cpu'               # (Optional) Specify torch device
)
```

### Key Features of DatasetRNN

1. **Feature Normalization**:
   - Specify which feature dimensions to normalize using `normalize_features`
   - Normalization is performed per feature to range [0, 1]

2. **Sequence Processing**:
   - Optional fixed sequence length with `sequence_length`
   - Configurable stride for sequence creation
   - Automatic handling of variable-length sequences

## 5. Splitting Datasets

SPICE provides two main methods for splitting datasets for training and testing:

### Splitting Along Time Dimension

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

### Splitting Along Session Dimension

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

## Next Steps

- Learn how to [train a basic Rescorla-Wagner model](rescorla_wagner.html)
- Understand how to [incorporate custom mechanisms such as forgetting](rescorla_wagner_forgetting.html)
- Explore [working with hardcoded equations](hardcoded_equations.html)


```python

```
