---
layout: default
title: Quick Start
nav_order: 3
---

# Quick Start Guide

This guide will help you get started with SPICE by walking through a complete example using a two-armed bandit task.

## Basic Usage

Here's a complete example that demonstrates the core functionality of SPICE:

```python
from spice.estimator import SpiceEstimator
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset

# Step 1: Create a simulated environment
environment = BanditsDrift(
    sigma=0.2,  # Noise level in the environment
    n_actions=2  # Number of possible actions (arms)
)

# Step 2: Create a Q-learning agent
agent = AgentQ(
    n_actions=2,
    alpha_reward=0.6,   # Learning rate for positive rewards
    alpha_penalty=0.6,  # Learning rate for negative rewards
    forget_rate=0.3,    # Rate at which the agent forgets previous learning
)

# Step 3: Generate synthetic data
dataset, _, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=200,    # Number of trials per session
    n_sessions=256,  # Number of sessions to simulate
)

# Step 4: Create and configure SPICE estimator
spice_estimator = SpiceEstimator(
    rnn_class=RescorlaWagnerRNN,           # Type of RNN to use
    spice_config=RESCOLA_WAGNER_CONFIG,     # Configuration for SPICE
    hidden_size=8,                          # Size of hidden layer
    learning_rate=5e-3,                     # Learning rate for training
    epochs=16,                              # Number of training epochs
    n_steps_per_call=16,                    # Steps per training iteration
    spice_participant_id=0,                 # Participant ID for analysis
    verbose=True,                           # Enable progress output
)

# Step 5: Train the model
spice_estimator.fit(dataset.xs, dataset.ys)

# Step 6: Extract learned features
features = spice_estimator.spice_agent.get_spice_features()
for id, feat in features.items():
    print(f"\nAgent {id}:")
    for model_name, (feat_names, coeffs) in feat.items():
        print(f"  {model_name}:")
        for name, coeff in zip(feat_names, coeffs):
            print(f"    {name}: {coeff}")

# Step 7: Make predictions
pred_rnn, pred_spice = spice_estimator.predict(dataset.xs)
```

## Understanding the Components

### Environment Setup
The `BanditsDrift` class creates a two-armed bandit environment where:
- Each arm has a reward probability that drifts over time
- `sigma` controls the noise level in the environment
- `n_actions` specifies the number of possible actions (arms)

### Agent Configuration
The `AgentQ` class implements a Q-learning agent with:
- Separate learning rates for rewards and penalties
- A forgetting mechanism that gradually decays learned values
- Support for multiple actions

### SPICE Estimator
The `SpiceEstimator` class is the main interface to SPICE, combining:
- An RNN for learning behavioral patterns
- SINDy for discovering interpretable equations
- Scikit-learn compatible interface

## Working with Real Data

When working with real data instead of simulated data, your input should be structured as:

```python
# X: Input features (trials × features)
# y: Target variables (trials × targets)
spice_estimator.fit(X, y)
```

The exact structure of X and y will depend on your specific task.

## Next Steps

Now that you've seen the basics, you might want to:

1. Check out the [tutorials](tutorials.html) for more complex examples
2. Read the [API documentation](api.html) for detailed information
3. Learn about [customizing SPICE](customization.html) for your specific needs

## Common Patterns

Here are some common patterns you might find useful:

### Custom RNN Architecture
```python
from spice.estimator import SpiceEstimator
from your_module import CustomRNN, CUSTOM_CONFIG

estimator = SpiceEstimator(
    rnn_class=CustomRNN,
    spice_config=CUSTOM_CONFIG,
    # ... other parameters
)
```

### Multiple Participants
```python
# Train on multiple participants
for participant_id in range(n_participants):
    estimator = SpiceEstimator(
        spice_participant_id=participant_id,
        # ... other parameters
    )
    estimator.fit(X[participant_id], y[participant_id])
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_size': [4, 8, 16],
    'learning_rate': [1e-3, 5e-3, 1e-2],
}

grid_search = GridSearchCV(
    estimator=SpiceEstimator(),
    param_grid=param_grid,
    cv=5
)
``` 