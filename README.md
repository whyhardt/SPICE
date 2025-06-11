# Computational Discovery of Sparse and Interpretable Cognitive Equations (SPICE)

![SPICE Logo](https://github.com/whyhardt/SPICE/blob/main/figures/spice_logo.png?raw=true)

SPICE is a framework for automating scientific practice in cognitive science and is based on a two cornerstones:

1. A task-specific RNN is trained to predict human behavior and thus learn implicitly latent cognitive mechanisms.

2. Sparse Identification of nonlinear Dynamics (SINDy; an equation discovery algorithm) is used to obtain mathematically interpretable equations for the learned cognitive mechanisms.

📚 **Documentation**: [https://whyhardt.github.io/SPICE/](https://whyhardt.github.io/SPICE/)

The resulting model with the neural-network architecture but with equations instead of RNN modules is called SPICE model. An overview is given in Figure 1.

This README file gives an overview on how to install and run SPICE as a scikit-learn estimator. To learn how to use SPICE in more comprehensive scenarios, you can go to [tutorials](tutorials).

![Figure 1 - SPICE Overview](https://github.com/whyhardt/SPICE/blob/main/figures/spice_overview.jpg?raw=true "Figure 1: SPICE overview")

## Installation

You can install SPICE using pip:

```bash
pip install autospice
```

or, you can clone this repository and install it locally from the root folder:

```bash
pip install -e .
```

## Features

- Scikit-learn compatible estimator interface
- Customizable network architecture for identifying complex cognitive mechanisms
- Participant embeddings for identifying individual differences
- Precoded models for simple Rescorla-Wagner, forgetting mechanism, choise perseveration and parcitipant embeddings

## Quick Start

```python
from spice.estimator import SpiceEstimator
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset

# Simulate dataset from a two-armed bandit task with a Q agent
environment = BanditsDrift(sigma=0.2, n_actions=2)

agent = AgentQ(
    n_actions=2,
    alpha_reward=0.6,  # Learning rate for positive rewards 
    alpha_penalty=0.6,  # Learning rate for negative rewards
    forget_rate=0.3,
)

dataset, _, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=200,
    n_sessions=256,
)

# Create and fit SPICE model
spice_estimator = SpiceEstimator(
    rnn_class=RescorlaWagnerRNN,
    spice_config=RESCOLA_WAGNER_CONFIG,
    hidden_size=8,
    learning_rate=5e-3,
    epochs=16,
    n_steps_per_call=16,
    spice_participant_id=0,
    save_path_rnn='rnn_model.pkl',   # (Optional) File path (.pkl) to save RNN model after training
    save_path_spice='spice_model.pkl', # (Optional) File path (.pkl) to save SPICE model after training
    verbose=True,
)

spice_estimator.fit(dataset.xs, dataset.ys)

# Get learned SPICE features
features = spice_estimator.spice_agent.get_spice_features()
for id, feat in features.items():
    print(f"\nAgent {id}:")
    for model_name, (feat_names, coeffs) in feat.items():
        print(f"  {model_name}:")
        for name, coeff in zip(feat_names, coeffs):
            print(f"    {name}: {coeff}")

# Predict behavior
pred_rnn, pred_spice = spice_estimator.predict(dataset.xs)

print("\nPrediction shapes:")
print(f"RNN predictions: {pred_rnn.shape}")
print(f"SPICE predictions: {pred_spice.shape}")  
```

### Saving and loading models

In addition to specifying save paths when creating the SpiceEstimator, you can also save an existing estimator's models, or load from save files:

```python
# Save trained model to file
spice_estimator.save_spice(path_rnn='rnn_model.pkl', path_spice='spice_model.pkl')

# Load saved model
loaded_spice = SpiceEstimator.load_spice(path_rnn='rnn_model.pkl', path_spice='spice_model.pkl')

# Use loaded model for predictions
pred_rnn, pred_spice = loaded_spice.predict(dataset.xs)
```

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spice2025,
  title = {SPICE: Sparse and Interpretable Cognitive Equations},
  year = {2025},
  author = {Weinhardt, Daniel},
  url = {https://github.com/whyhardt/SPICE}
}
```
