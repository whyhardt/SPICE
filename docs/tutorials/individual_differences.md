---
layout: default
title: Modeling Individual Differences
parent: Tutorials
nav_order: 4
---

# Modeling Individual Differences in SPICE

This tutorial explains how to model and analyze individual differences in cognitive mechanisms using SPICE. You'll learn how to:
- Use participant embeddings to capture individual differences
- Train models with participant-specific parameters
- Analyze and interpret individual differences in cognitive mechanisms

## Prerequisites

Before starting this tutorial, make sure you have:
- Completed the previous tutorials
- Understanding of basic reinforcement learning concepts
- Familiarity with embedding spaces and individual differences

## Why Model Individual Differences?

People differ in how they learn and make decisions. These differences can manifest in:
- Learning rates
- Decision noise (inverse temperature)
- Forgetting rates
- Choice perseveration
- Strategy preferences

SPICE can capture these differences through participant embeddings, allowing us to:
1. Model participant-specific cognitive mechanisms
2. Discover patterns in individual differences
3. Make personalized predictions

## Tutorial Contents

1. Setting up participant embeddings
2. Training models with individual differences
3. Analyzing participant-specific parameters
4. Visualizing individual differences
5. Best practices for individual difference modeling

## Interactive Version

This is the static web version of the tutorial. For an interactive version:

1. Go to the SPICE repository
2. Navigate to `tutorials/4_individual_differences.ipynb`
3. Run the notebook in Jupyter

## Full Tutorial

[View or download the complete notebook](https://github.com/whyhardt/SPICE/blob/main/tutorials/4_individual_differences.ipynb)

---

## Step-by-Step Guide

### 1. Setup and Imports

```python
import numpy as np
import torch
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset
from spice.precoded import ParticipantEmbeddingRNN, PARTICIPANT_EMBEDDING_RNN_CONFIG
from spice.estimator import SpiceEstimator

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
```

### 2. Create Environment and Multiple Agents

We'll simulate data from multiple agents with different parameters:

```python
# Set up the environment
n_actions = 2
sigma = 0.2
environment = BanditsDrift(sigma=sigma, n_actions=n_actions)

# Create multiple agents with different parameters
n_participants = 50
agents = []
for _ in range(n_participants):
    agent = AgentQ(
        n_actions=n_actions,
        alpha_reward=np.random.uniform(0.2, 0.8),    # Random learning rate
        alpha_penalty=np.random.uniform(0.2, 0.8),   # Random penalty learning rate
        forget_rate=np.random.uniform(0.1, 0.5),     # Random forgetting rate
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
    'xs': torch.cat([d.xs for d in datasets], dim=1),  # Combine across participants
    'ys': torch.cat([d.ys for d in datasets], dim=1)
}
```

### 3. Using the Participant Embedding RNN

SPICE provides a precoded RNN that includes participant embeddings:

```python
# Create and train SPICE model with participant embeddings
spice_estimator = SpiceEstimator(
    rnn_class=ParticipantEmbeddingRNN,
    spice_config=PARTICIPANT_EMBEDDING_RNN_CONFIG,
    hidden_size=8,
    learning_rate=5e-3,
    epochs=16,
    n_participants=n_participants,  # Specify number of participants
    verbose=True
)

spice_estimator.fit(combined_dataset.xs, combined_dataset.ys)
```

### 4. Analyzing Individual Differences

Extract and examine the participant embeddings:

```python
# Get participant embeddings
embeddings = spice_estimator.get_participant_embeddings()
for participant_id, embedding in embeddings.items():
    print(f"Participant {participant_id} embedding:", embedding)

# Get learned features for each participant
features = spice_estimator.get_spice_features()
for id, feat in features.items():
    print(f"\nParticipant {id}:")
    for model_name, (feat_names, coeffs) in feat.items():
        print(f"  {model_name}:")
        for name, coeff in zip(feat_names, coeffs):
            print(f"    {name}: {coeff}")
```

### 5. Visualizing Individual Differences

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Convert embeddings to numpy array for visualization
embedding_matrix = np.stack([emb.detach().numpy() for emb in embeddings.values()])

# PCA visualization of participant embeddings
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(embedding_matrix)

plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
plt.title('Participant Embeddings (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

## Understanding the Results

When analyzing individual differences, look for:

1. **Clustering**: Groups of participants with similar cognitive mechanisms
2. **Parameter Distributions**: How cognitive parameters vary across participants
3. **Strategy Differences**: Different approaches to the same task
4. **Learning Trajectories**: How learning rates and strategies evolve

## Best Practices

When modeling individual differences:

1. **Data Collection**
   - Ensure sufficient trials per participant
   - Balance participant characteristics
   - Consider task complexity

2. **Model Design**
   - Choose appropriate embedding dimensions
   - Consider regularization for embeddings
   - Balance model complexity with data size

3. **Analysis**
   - Validate individual predictions
   - Look for meaningful patterns
   - Consider real-world implications

## Next Steps

After completing this tutorial, you can:
1. Apply individual difference modeling to your own data
2. Explore more complex embedding architectures
3. Move on to [Weinhardt et al. 2024 Case Study](weinhardt_2024.html)

## Common Issues and Solutions

- **Overfitting**: Use dropout and regularization for embeddings
- **High Variance**: Increase trials per participant or reduce embedding dimension
- **Poor Generalization**: Balance model complexity with data size

## Additional Resources

- [Individual Differences in RL](https://www.nature.com/articles/s41593-020-0658-y)
- [SPICE API Documentation](../api.html)
- [GitHub Repository](https://github.com/whyhardt/SPICE) 