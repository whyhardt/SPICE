---
layout: default
title: API Reference
nav_order: 4
---

# API Reference

This page documents the main classes and functions in SPICE.

## Core Classes

### SpiceEstimator

The main class for training and using SPICE models. Implements scikit-learn's estimator interface.

```python
from spice.estimator import SpiceEstimator
```

#### Parameters

- `rnn_class` (BaseRNN): RNN class to use (can be precoded or custom implementation)
- `spice_config` (SpiceConfig): Configuration for SPICE features and library
- `hidden_size` (int, default=8): Size of RNN hidden layer
- `dropout` (float, default=0.25): Dropout rate for RNN
- `n_actions` (int, default=2): Number of possible actions
- `n_participants` (int, default=0): Number of participants
- `n_experiments` (int, default=0): Number of experiments
- `epochs` (int, default=128): Number of training epochs
- `learning_rate` (float, default=5e-3): Learning rate for training
- `spice_optim_threshold` (float, default=0.03): Threshold for SPICE optimization
- `spice_participant_id` (int, optional): ID of specific participant to analyze
- `verbose` (bool, default=False): Whether to print progress information

#### Methods

##### fit(conditions, targets)
Trains both RNN and SPICE models on given data.

```python
def fit(conditions: np.ndarray, targets: np.ndarray)
"""
Args:
    conditions: Array of shape (n_participants, n_trials, n_features)
    targets: Array of shape (n_participants, n_trials, n_actions)
"""
```

##### predict(conditions)
Makes predictions using both RNN and SPICE models.

```python
def predict(conditions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
"""
Args:
    conditions: Array of shape (n_participants, n_trials, n_features)
Returns:
    Tuple containing:
    - RNN predictions
    - SPICE predictions
"""
```

##### get_spice_features()
Returns learned SPICE features and equations.

```python
def get_spice_features() -> Dict
"""
Returns:
    Dictionary mapping participant IDs to their learned features and equations
"""
```

### SpiceConfig

Configuration class for setting up SPICE models.

```python
from spice.estimator import SpiceConfig
```

#### Parameters

- `library_setup` (Dict[str, List[str]]): Maps features to library components
- `filter_setup` (Dict[str, List]): Maps features to filter conditions
- `control_parameters` (List[str]): List of control parameter names
- `rnn_modules` (List[str]): List of RNN module names

## Precoded Models

SPICE comes with several precoded RNN models for common cognitive mechanisms:

### RescorlaWagnerRNN

Implementation of the Rescorla-Wagner learning model.

```python
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG
```

### ForgettingRNN

Model incorporating forgetting mechanisms.

```python
from spice.precoded import ForgettingRNN
```

### LearningRateRNN

Model with adaptive learning rates.

```python
from spice.precoded import LearningRateRNN
```

### ParticipantEmbeddingRNN

Model that learns participant-specific embeddings.

```python
from spice.precoded import ParticipantEmbeddingRNN
```

## Agents

### AgentSpice

SPICE agent that combines RNN and SINDy equations.

```python
from spice.resources.bandits import AgentSpice
```

#### Methods

##### get_spice_features()
Extracts features and coefficients for each module and participant.

```python
def get_spice_features(mapping_modules_values: dict = None) -> Dict[int, Dict]
"""
Args:
    mapping_modules_values: Optional mapping of modules to memory state values
Returns:
    Dictionary mapping participant IDs to their features and coefficients
"""
```

##### count_parameters()
Counts non-zero parameters for each participant.

```python
def count_parameters(mapping_modules_values: dict = None) -> Dict[int, int]
"""
Args:
    mapping_modules_values: Optional mapping of modules to memory state values
Returns:
    Dictionary mapping participant IDs to parameter counts
"""
```

## Utility Functions

### fit_spice()

Fits SPICE by replacing RNN modules with SINDy equations.

```python
from spice.resources.sindy_training import fit_spice

def fit_spice(
    rnn_modules: List[str],
    control_signals: List[str],
    agent_rnn: AgentNetwork,
    data: DatasetRNN = None,
    polynomial_degree: int = 2,
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 0.1,
    participant_id: int = None,
    verbose: bool = False
) -> Tuple[AgentSpice, float]
"""
Args:
    rnn_modules: List of RNN module names to replace
    control_signals: List of control signal names
    agent_rnn: Trained RNN agent
    data: Training dataset
    polynomial_degree: Degree for polynomial features
    optimizer_threshold: Threshold for optimization
    optimizer_alpha: Alpha parameter for optimization
    participant_id: Specific participant to process
    verbose: Whether to print progress
Returns:
    Tuple of (SPICE agent, loss value)
"""
```

### optimize_for_participant()

Optimizes SPICE parameters for a specific participant.

```python
from spice.resources.optimizer_selection import optimize_for_participant

def optimize_for_participant(
    participant_id: int,
    agent_rnn: AgentNetwork,
    data: DatasetRNN,
    metric_rnn: float,
    rnn_modules: list,
    control_signals: list,
    library_setup: dict,
    filter_setup: dict,
    polynomial_degree: int,
    n_sessions_off_policy: int,
    n_trials_optuna: int = 50,
    verbose: bool = False
)
"""
Args:
    participant_id: ID of participant to optimize for
    agent_rnn: Trained RNN agent
    data: Training data
    metric_rnn: RNN performance metric
    rnn_modules: List of RNN modules
    control_signals: List of control signals
    library_setup: Library configuration
    filter_setup: Filter configuration
    polynomial_degree: Degree for polynomial features
    n_sessions_off_policy: Number of off-policy sessions
    n_trials_optuna: Number of optimization trials
    verbose: Whether to print progress
"""
``` 