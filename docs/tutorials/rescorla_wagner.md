---
layout: default
title: Basic Rescorla-Wagner Model
parent: Tutorials
nav_order: 2
---

# Basic Rescorla-Wagner Model Tutorial

This tutorial introduces SPICE using a simple Rescorla-Wagner learning model. You'll learn how to:
- Set up a basic SPICE model
- Train it on simulated data
- Extract and interpret the discovered equations

## Prerequisites

Before starting this tutorial, make sure you have:
- SPICE installed (`pip install autospice`)
- Basic understanding of reinforcement learning
- Familiarity with Python and NumPy

## The Rescorla-Wagner Model

The Rescorla-Wagner model is a fundamental model of associative learning that describes how associations between stimuli and outcomes are learned through experience. The basic equation is:

ΔV = α(λ - V)

where:
- V is the associative strength
- α is the learning rate
- λ is the maximum possible associative strength
- ΔV is the change in associative strength

## Tutorial Contents

1. Setting up the environment
2. Creating simulated data
3. Training the SPICE model
4. Analyzing the results
5. Interpreting the equations

## Interactive Version

This is the static web version of the tutorial. For an interactive version:

1. Go to the SPICE repository
2. Navigate to `tutorials/1_rescorla_wagner.ipynb`
3. Run the notebook in Jupyter

## Full Tutorial

[View or download the complete notebook](https://github.com/whyhardt/SPICE/blob/main/tutorials/1_rescorla_wagner.ipynb)

---

## Step-by-Step Guide

### 1. Setup

First, let's import the necessary modules:

```python
from spice.estimator import SpiceEstimator
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset
import numpy as np
```

### 2. Create the Environment

We'll create a two-armed bandit environment:

```python
environment = BanditsDrift(
    sigma=0.2,  # Noise level
    n_actions=2  # Number of arms
)
```

### 3. Create the Agent

Set up a Q-learning agent with specific learning parameters:

```python
agent = AgentQ(
    n_actions=2,
    alpha_reward=0.6,   # Learning rate for rewards
    alpha_penalty=0.6,  # Learning rate for penalties
    forget_rate=0.3,    # Rate of forgetting
)
```

### 4. Generate Data

Create a synthetic dataset for training:

```python
dataset, _, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=200,    # Trials per session
    n_sessions=256,  # Number of sessions
)
```

### 5. Create and Train SPICE Model

Set up and train the SPICE model:

```python
spice_estimator = SpiceEstimator(
    rnn_class=RescorlaWagnerRNN,
    spice_config=RESCOLA_WAGNER_CONFIG,
    hidden_size=8,
    learning_rate=5e-3,
    epochs=16,
    verbose=True
)

spice_estimator.fit(dataset.xs, dataset.ys)
```

### 6. Extract Learned Features

Examine what SPICE has learned:

```python
features = spice_estimator.spice_agent.get_spice_features()
for id, feat in features.items():
    print(f"\nAgent {id}:")
    for model_name, (feat_names, coeffs) in feat.items():
        print(f"  {model_name}:")
        for name, coeff in zip(feat_names, coeffs):
            print(f"    {name}: {coeff}")
```

### 7. Make Predictions

Use the trained model to make predictions:

```python
pred_rnn, pred_spice = spice_estimator.predict(dataset.xs)
```

## Understanding the Results

The SPICE model should discover equations similar to the Rescorla-Wagner update rule. Key things to look for:

1. The relationship between reward prediction error and value updates
2. The learning rate parameter
3. How well the discovered equations match the original agent's parameters

## Next Steps

After completing this tutorial, you can:
1. Experiment with different parameter values
2. Try more complex environments
3. Move on to the [Rescorla-Wagner with Forgetting](rescorla_wagner_forgetting.html) tutorial

## Common Issues and Solutions

- **Poor Convergence**: Try increasing the number of epochs or adjusting the learning rate
- **Overfitting**: Reduce the hidden size or increase the dataset size
- **Unstable Training**: Adjust the optimizer parameters or reduce the learning rate

## Additional Resources

- [SPICE API Documentation](../api.html)
- [GitHub Repository](https://github.com/whyhardt/SPICE) 