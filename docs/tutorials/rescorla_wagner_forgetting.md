---
layout: default
title: Rescorla-Wagner with Forgetting
parent: Tutorials
nav_order: 2
---

# Rescorla-Wagner with Forgetting Tutorial

This tutorial extends the basic Rescorla-Wagner model by adding a forgetting mechanism for not-chosen actions. You'll learn how to:
- Implement forgetting mechanisms in SPICE
- Work with multiple cognitive mechanisms simultaneously
- Understand how SPICE discovers interaction effects

## Prerequisites

Before starting this tutorial, make sure you have:
- Completed the [Basic Rescorla-Wagner Tutorial](rescorla_wagner.html)
- SPICE installed with all dependencies
- Understanding of basic reinforcement learning concepts

## The Forgetting Mechanism

In real-world learning scenarios, humans tend to forget information about options they haven't chosen recently. The forgetting mechanism models this by:
- Gradually decreasing the value of non-chosen actions
- Maintaining separate learning rates for chosen and non-chosen actions
- Allowing for dynamic adjustment of forgetting rates

## Tutorial Contents

1. Setting up the environment with forgetting
2. Creating a Q-learning agent with forgetting
3. Training SPICE with multiple mechanisms
4. Analyzing the discovered equations
5. Implementing custom forgetting mechanisms

## Interactive Version

This is the static web version of the tutorial. For an interactive version:

1. Go to the SPICE repository
2. Navigate to `tutorials/2_rescorla-wagner_forgetting.ipynb`
3. Run the notebook in Jupyter

## Full Tutorial

[View or download the complete notebook](https://github.com/whyhardt/SPICE/blob/main/tutorials/2_rescorla-wagner_forgetting.ipynb)

---

## Step-by-Step Guide

### 1. Setup and Imports

```python
import numpy as np
import torch
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
```

### 2. Create Environment and Agent

Now we'll create an agent with forgetting:

```python
# Set up the environment
n_actions = 2
sigma = 0.2
environment = BanditsDrift(sigma=sigma, n_actions=n_actions)

# Set up the agent with forgetting
agent = AgentQ(
    n_actions=n_actions,
    alpha_reward=0.3,    # Learning rate for rewards
    forget_rate=0.2,     # Rate of forgetting for non-chosen actions
)

# Generate dataset
n_trials = 200
n_sessions = 256
dataset, _, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=n_trials,
    n_sessions=n_sessions,
)
```

### 3. Using the Precoded Forgetting RNN

SPICE provides a precoded RNN that includes forgetting mechanisms:

```python
from spice.precoded import ForgettingRNN, FORGETTING_CONFIG
from spice.estimator import SpiceEstimator

# Create and train SPICE model
spice_estimator = SpiceEstimator(
    rnn_class=ForgettingRNN,
    spice_config=FORGETTING_CONFIG,
    hidden_size=8,
    learning_rate=5e-3,
    epochs=16,
    verbose=True
)

spice_estimator.fit(dataset.xs, dataset.ys)
```

### 4. Analyzing the Results

Extract and examine the learned features:

```python
features = spice_estimator.spice_agent.get_spice_features()
for id, feat in features.items():
    print(f"\nAgent {id}:")
    for model_name, (feat_names, coeffs) in feat.items():
        print(f"  {model_name}:")
        for name, coeff in zip(feat_names, coeffs):
            print(f"    {name}: {coeff}")
```

### 5. Custom Forgetting Mechanisms

You can also implement your own forgetting mechanism:

```python
from spice.estimator import SpiceConfig

CUSTOM_FORGETTING_CONFIG = SpiceConfig(
    library_setup={
        'x_value_reward': ['c_reward'],
        'x_value_forget': ['c_action'],
    },
    filter_setup={
        'x_value_reward': ['c_action', 1, True],
        'x_value_forget': ['c_action', 0, True],
    },
    control_parameters=['c_action', 'c_reward'],
    rnn_modules=['x_value_reward', 'x_value_forget']
)
```

## Understanding the Results

When analyzing the results, look for:

1. **Forgetting Rate**: The coefficient that determines how quickly non-chosen values decay
2. **Interaction Effects**: How forgetting interacts with reward learning
3. **Value Updates**: Different update rules for chosen vs non-chosen actions

## Common Patterns

The model typically discovers:
- Faster learning rates for chosen actions
- Gradual decay for non-chosen actions
- Balance between exploration and exploitation

## Next Steps

After completing this tutorial, you can:
1. Experiment with different forgetting rates
2. Implement more complex forgetting mechanisms
3. Move on to [Working with Hardcoded Equations](hardcoded_equations.html)

## Common Issues and Solutions

- **Unstable Learning**: Try reducing the learning rate or increasing batch size
- **Poor Forgetting**: Adjust the forgetting rate or increase training data
- **Convergence Issues**: Increase the number of epochs or adjust optimizer parameters

## Additional Resources

- [Memory and Forgetting in RL](https://arxiv.org/abs/1602.02445)
- [SPICE API Documentation](../api.html)
- [GitHub Repository](https://github.com/whyhardt/SPICE) 