---
layout: default
title: Weinhardt et al. 2024 Case Study
parent: Tutorials
nav_order: 6
---

# Weinhardt et al. 2024 Case Study

This tutorial implements the model from Weinhardt et al. (2024) paper "[Computational Discovery of Cognitive Dynamics](https://openreview.net/forum?id=x2WDZrpgmB)". You'll learn how to:
- Implement a complex cognitive model combining multiple mechanisms
- Work with both goal-directed and non-goal-directed behavior
- Model choice perseveration bias
- Combine RNN modules with hardcoded equations

## Prerequisites

Before starting this tutorial, make sure you have:
- Completed all previous tutorials
- Understanding of reinforcement learning and choice behavior
- Familiarity with participant embeddings and individual differences

## The Weinhardt 2024 Model

The model combines two key components:
1. Goal-directed behavior (`x_value_reward`)
   - Learning from rewards
   - Value-based decision making
   
2. Non-goal-directed behavior (`x_value_choice`)
   - Choice perseveration bias
   - Previous action influence
   - Habit formation

## Tutorial Contents

1. Setting up the model architecture
2. Implementing reward and choice mechanisms
3. Combining multiple cognitive processes
4. Training and analyzing the model
5. Understanding the results

## Interactive Version

This is the static web version of the tutorial. For an interactive version:

1. Go to the SPICE repository
2. Navigate to `tutorials/5_weinhardt_et_al_2024.ipynb`
3. Run the notebook in Jupyter

## Full Tutorial

[View or download the complete notebook](https://github.com/whyhardt/SPICE/blob/main/tutorials/5_weinhardt_et_al_2024.ipynb)

---

## Step-by-Step Guide

### 1. Setup and Imports

```python
import numpy as np
import torch
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset
from spice.precoded import Weinhardt2024RNN, WEINHARDT_2024_CONFIG
from spice.estimator import SpiceEstimator

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
```

### 2. Create Environment and Agents

We'll create agents that exhibit both goal-directed and habitual behavior:

```python
# Set up the environment
n_actions = 2
sigma = 0.2
environment = BanditsDrift(sigma=sigma, n_actions=n_actions)

# Create agents with different parameters
n_participants = 50
agents = []
for _ in range(n_participants):
    agent = AgentQ(
        n_actions=n_actions,
        alpha_reward=np.random.uniform(0.2, 0.8),    # Learning rate for rewards
        alpha_penalty=np.random.uniform(0.2, 0.8),   # Learning rate for penalties
        forget_rate=np.random.uniform(0.1, 0.5),     # Forgetting rate
        beta_choice=np.random.uniform(0.5, 2.0),     # Choice perseveration strength
    )
    agents.append(agent)

# Generate dataset for each agent
datasets = []
for agent in agents:
    dataset, _, _ = create_dataset(
        agent=agent,
        environment=environment,
        n_trials=200,
        n_sessions=1,
    )
    datasets.append(dataset)

# Combine datasets
combined_dataset = {
    'xs': torch.cat([d.xs for d in datasets], dim=1),
    'ys': torch.cat([d.ys for d in datasets], dim=1)
}
```

### 3. Using the Weinhardt 2024 RNN

The model includes both RNN modules and hardcoded equations:

```python
# Create and train SPICE model
spice_estimator = SpiceEstimator(
    rnn_class=Weinhardt2024RNN,
    spice_config=WEINHARDT_2024_CONFIG,
    hidden_size=8,
    learning_rate=5e-3,
    epochs=16,
    n_participants=n_participants,
    dropout=0.25,  # Added dropout for regularization
    verbose=True
)

spice_estimator.fit(combined_dataset.xs, combined_dataset.ys)
```

### 4. Analyzing Model Components

Extract and examine the learned mechanisms:

```python
# Get participant embeddings
embeddings = spice_estimator.get_participant_embeddings()

# Get learned features
features = spice_estimator.get_spice_features()
for id, feat in features.items():
    print(f"\nParticipant {id}:")
    for model_name, (feat_names, coeffs) in feat.items():
        print(f"  {model_name}:")
        for name, coeff in zip(feat_names, coeffs):
            print(f"    {name}: {coeff}")

# Analyze reward vs choice influence
def analyze_value_components(features):
    reward_coeffs = []
    choice_coeffs = []
    for id, feat in features.items():
        for model_name, (feat_names, coeffs) in feat.items():
            if 'value_reward' in model_name:
                reward_coeffs.extend(coeffs)
            elif 'value_choice' in model_name:
                choice_coeffs.extend(coeffs)
    return np.mean(reward_coeffs), np.mean(choice_coeffs)

reward_influence, choice_influence = analyze_value_components(features)
print(f"\nAverage reward influence: {reward_influence:.3f}")
print(f"Average choice influence: {choice_influence:.3f}")
```

### 5. Model Architecture Details

The Weinhardt 2024 model includes:

1. **Memory States**
   ```python
   init_values = {
       'x_value_reward': 0.5,    # Reward-based value
       'x_value_choice': 0.0,    # Choice-based value
       'x_learning_rate_reward': 0.0,  # Dynamic learning rate
   }
   ```

2. **RNN Modules**
   ```python
   # Learning rate module
   self.submodules_rnn['x_learning_rate_reward'] = self.setup_module(
       input_size=2+self.embedding_size, 
       dropout=0.25
   )
   
   # Value update modules
   self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(
       input_size=0+self.embedding_size, 
       dropout=0.25
   )
   self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(
       input_size=0+self.embedding_size, 
       dropout=0.25
   )
   ```

3. **Hardcoded Equations**
   ```python
   # Reward prediction error update
   self.submodules_eq['x_value_reward_chosen'] = lambda value, inputs: (
       value + inputs[..., 1] * (inputs[..., 0] - value)
   )
   ```

## Understanding the Results

When analyzing the model, look for:

1. **Balance of Mechanisms**
   - Relative influence of reward vs choice values
   - Individual differences in mechanism reliance
   - Learning rate adaptation

2. **Choice Perseveration**
   - Strength of habit formation
   - Impact of previous choices
   - Individual variation in perseveration

3. **Learning Dynamics**
   - Interaction between mechanisms
   - Adaptation to environment changes
   - Strategy shifts over time

## Best Practices

When working with complex models:

1. **Model Implementation**
   - Carefully balance components
   - Use appropriate regularization
   - Monitor mechanism interactions

2. **Training**
   - Start with simpler versions
   - Validate each component
   - Use sufficient data

3. **Analysis**
   - Examine mechanism contributions
   - Look for emergent behaviors
   - Consider individual differences

## Next Steps

After completing this tutorial, you can:
1. Modify the model for your research
2. Add new cognitive mechanisms
3. Apply to different experimental paradigms

## Common Issues and Solutions

- **Component Dominance**: Adjust scaling factors or learning rates
- **Training Instability**: Reduce learning rate or add regularization
- **Poor Generalization**: Increase dropout or data size

## Additional Resources

- [Original Paper](https://openreview.net/forum?id=x2WDZrpgmB)
- [SPICE API Documentation](../api.html)
- [GitHub Repository](https://github.com/whyhardt/SPICE) 