---
layout: default
title: Working with Hardcoded Equations
parent: Tutorials
nav_order: 3
---

# Working with Hardcoded Equations

This tutorial explains how to use predefined equations in SPICE models. You'll learn how to:
- Incorporate known cognitive mechanisms as hardcoded equations
- Combine hardcoded equations with learned mechanisms
- Optimize model performance using domain knowledge

## Prerequisites

Before starting this tutorial, make sure you have:
- Completed the previous tutorials
- Understanding of basic cognitive modeling equations
- Familiarity with the SPICE architecture

## Why Use Hardcoded Equations?

Sometimes we have strong theoretical knowledge about certain cognitive mechanisms. For example:
- The reward prediction error in reinforcement learning
- Memory decay functions in forgetting models
- Attention mechanisms in decision making

Using hardcoded equations allows us to:
1. Incorporate established theoretical knowledge
2. Reduce the search space for model discovery
3. Focus learning on unknown mechanisms

## Tutorial Contents

1. Understanding when to use hardcoded equations
2. Implementing reward prediction error as a hardcoded module
3. Combining hardcoded and learned mechanisms
4. Analyzing model performance
5. Best practices for equation design

## Interactive Version

This is the static web version of the tutorial. For an interactive version:

1. Go to the SPICE repository
2. Navigate to `tutorials/3_hardcoded_equations.ipynb`
3. Run the notebook in Jupyter

## Full Tutorial

[View or download the complete notebook](https://github.com/whyhardt/SPICE/blob/main/tutorials/3_hardcoded_equations.ipynb)

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

```python
# Set up the environment
n_actions = 2
sigma = 0.2
environment = BanditsDrift(sigma=sigma, n_actions=n_actions)

# Set up the agent
agent = AgentQ(
    n_actions=n_actions,
    alpha_reward=0.6,   # Learning rate for positive rewards
    alpha_penalty=0.6,  # Learning rate for negative rewards
    forget_rate=0.3,
)

# Generate dataset
dataset, _, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=200,
    n_sessions=256,
)
```

### 3. Using Precoded Models with Hardcoded Equations

SPICE provides models with built-in hardcoded equations:

```python
from spice.precoded import LearningRateRNN, LEARNING_RATE_CONFIG
from spice.estimator import SpiceEstimator

# Create and train SPICE model
spice_estimator = SpiceEstimator(
    rnn_class=LearningRateRNN,
    spice_config=LEARNING_RATE_CONFIG,
    hidden_size=8,
    learning_rate=5e-3,
    epochs=16,
    verbose=True
)

spice_estimator.fit(dataset.xs, dataset.ys)
```

### 4. Creating Custom Hardcoded Equations

You can create your own hardcoded equations:

```python
from spice.resources.rnn import BaseRNN
import torch.nn as nn

class CustomHardcodedRNN(BaseRNN):
    def __init__(self, n_actions, **kwargs):
        super().__init__(n_actions=n_actions, **kwargs)
        
        # Define hardcoded equation parameters
        self.alpha = nn.Parameter(torch.tensor(0.3))
        
    def forward_hardcoded(self, x_t, r_t):
        # Implement reward prediction error
        delta = r_t - x_t
        return self.alpha * delta
```

### 5. Combining Hardcoded and Learned Components

Create a configuration that uses both:

```python
from spice.estimator import SpiceConfig

MIXED_CONFIG = SpiceConfig(
    library_setup={
        'x_learning_rate': ['x_value', 'c_reward'],
    },
    filter_setup={
        'x_learning_rate': ['c_action', 1, True],
    },
    control_parameters=['c_action', 'c_reward'],
    rnn_modules=['x_learning_rate']
)
```

## Understanding the Results

When analyzing models with hardcoded equations, look for:

1. **Interaction Effects**: How hardcoded and learned mechanisms interact
2. **Parameter Adaptation**: How learned parameters modify hardcoded equations
3. **Model Performance**: Comparison with fully learned models

## Best Practices

When implementing hardcoded equations:

1. **Validate Assumptions**
   - Test the equations independently
   - Verify theoretical foundations
   - Compare with empirical data

2. **Balance Flexibility**
   - Allow some parameters to be learned
   - Don't over-constrain the model
   - Consider multiple theoretical accounts

3. **Document Clearly**
   - Explain equation choices
   - Reference theoretical sources
   - Document parameter meanings

## Next Steps

After completing this tutorial, you can:
1. Implement your own hardcoded equations
2. Combine multiple theoretical mechanisms
3. Move on to [Modeling Individual Differences](individual_differences.html)

## Common Issues and Solutions

- **Overly Rigid Models**: Allow some parameters to be learned
- **Poor Integration**: Ensure proper interaction between components
- **Numerical Instability**: Add bounds to hardcoded parameters

## Additional Resources

- [Reward Prediction Error Theory](https://www.nature.com/articles/nrn1406)
- [SPICE API Documentation](../api.html)
- [GitHub Repository](https://github.com/whyhardt/SPICE) 