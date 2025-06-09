# Computational Discovery of Sparse and Interpretable Cognitive Equations (SPICE)

![SPICE Logo](https://github.com/whyhardt/SPICE/blob/main/figures/spice_logo.png?raw=true)

SPICE is a framework for automating scientific practice in cognitive science and is based on a two cornerstones:

1. A task-specific RNN is trained to predict human behavior and thus learn implicitly latent cognitive mechanisms.

2. Sparse Identification of nonlinear Dynamics (SINDy; an equation discovery algorithm) is used to obtain mathematically interpretable equations for the learned cognitive mechanisms.

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
import numpy as np

# Create and configure the model
spice_estimator = SpiceEstimator(
    hidden_size=8,
    epochs=128,
    n_actions=2,
    n_participants=10,
    learning_rate=1e-2,
    sindy_optim_threshold=0.03,
    verbose=True
)

# Generate example data
conditions = np.random.rand(10, 100, 5)  # (n_participants, n_trials, n_features)
targets = np.random.randint(0, 2, size=(10, 100, 2))  # (n_participants, n_trials, n_actions)

# Fit the model
spice_estimator.fit(conditions, targets)

# Make predictions
pred_rnn, pred_sindy = model.predict(conditions)

# Get learned features
features = model.get_sindy_features()
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
