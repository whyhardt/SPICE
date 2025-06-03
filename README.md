# SPICE: RNN-SINDy Theorist

SPICE is a Python package that implements an RNN-SINDy theorist as a scikit-learn estimator for behavioral modeling. It combines Recurrent Neural Networks (RNN) for predicting behavioral choices with Sparse Identification of Nonlinear Dynamics (SINDy) for discovering underlying dynamical systems.

## Installation

You can install SPICE using pip:

```bash
pip install spice-rnn-sindy
```

## Features

- RNN-based behavioral prediction
- SINDy-based system identification
- Scikit-learn compatible estimator interface
- Support for multiple participants and experiments
- Customizable network architectures and training parameters

## Quick Start

```python
from spice import rnn_sindy_theorist
import numpy as np

# Create and configure the model
model = rnn_sindy_theorist(
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
model.fit(conditions, targets)

# Make predictions
pred_rnn, pred_sindy = model.predict(conditions)

# Get learned features
features = model.get_sindy_features()
```

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spice2024,
  title = {SPICE: RNN-SINDy Theorist for Behavioral Modeling},
  year = {2024},
  author = {SPICE Team},
  url = {https://github.com/yourusername/SPICE}
}
```