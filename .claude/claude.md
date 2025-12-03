# SPICE - Sparse and Interpretable Cognitive Equations

## Project Overview

SPICE is a framework for discovering symbolic cognitive mechanisms from behavioral data. It combines Recurrent Neural Networks (RNNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to extract interpretable mathematical equations that describe latent cognitive processes.

### Core Methodology

1. **RNN Training**: A task-specific RNN learns to predict human behavior, implicitly capturing latent cognitive mechanisms in disentangled submodules
2. **SINDy Regularization**: During training, SINDy equations act as regularizers (similar to SINDy-SHRED), pushing submodule dynamics toward spaces amenable to SINDy candidate terms
3. **Equation Discovery**: SINDy approximates the fitted dynamics in each disentangled submodule, yielding interpretable symbolic equations

## Repository Structure

```
SPICE/
├── spice/                      # Core framework (backend source code)
│   ├── estimator.py            # Scikit-learn compatible SpiceEstimator wrapper
│   ├── utils/
│   │   └── convert_dataset.py  # CSV to SpiceDataset conversion utilities
│   ├── resources/
│   │   ├── rnn.py              # PyTorch-based SPICE RNN architecture
│   │   ├── spice_training.py   # SPICE fitting procedures and training loops
│   │   ├── sindy_differentiable.py  # Differentiable PyTorch SINDy implementation
│   │   └── bandits.py          # Bandit task environments and agents
│   └── precoded/               # Pre-built model architectures
│       ├── choice.py           # Separate reward/choice value updates for chosen/unchosen actions
│       └── workingmemory.py    # Extends choice.py with working memory for past rewards/choices
│
├── weinhardt2025/              # Paper-specific code (fitting, benchmarking, analyses)
│   └── run.py                  # Main entry point for fitting SPICE to datasets
│
├── tutorials/                  # Usage tutorials and examples
├── requirements.txt            # Python dependencies
└── setup.py / pyproject.toml   # Package configuration
```

## Tech Stack

- **Language**: Python 3.x
- **ML Framework**: PyTorch
- **API Style**: Scikit-learn estimator interface
- **Package Name**: `autospice` (pip installable)

## Commands

```bash
# Installation
pip install autospice                    # Install from PyPI
pip install -e .                         # Install locally in editable mode

# Running Experiments
python weinhardt2025/run.py              # Fit SPICE model to dataset
```

## Data Format

### SpiceDataset Structure

The `SpiceDataset` contains two main components:

| Component | Shape | Description |
|-----------|-------|-------------|
| `xs` | `(batch, timesteps, features)` | Input conditions per trial |
| `ys` | `(batch, timesteps, actions)` | Next action (one-hot encoded) |

**Dimension details:**
- `batch` = `n_participants × n_experiments × n_blocks`
- `timesteps` = `n_choices_per_participant_per_block`
- `features` = actions (one-hot) + rewards per action (one-hot) + additional inputs + block_id + experiment_id + participant_id
- `actions` = one-hot encoded action space

### CSV Input Format

Use `spice/utils/convert_dataset.py` to convert CSV files to `SpiceDataset`. A typical CSV should contain all participants across all experiments and blocks with columns for:
- Participant ID, Experiment ID, Block number
- Actions taken (per trial)
- Rewards received (per action)
- Additional inputs (states, sensory input, etc.)

## Key Classes and Concepts

### SpiceEstimator (`spice/estimator.py`)
- Main user-facing class implementing sklearn's estimator interface
- Methods: `fit()`, `predict()`, `save_spice()`, `load_spice()`
- Wraps RNN training and SINDy equation discovery

### SpiceConfig
Configuration object carrying crucial information for interpretability:
- **Input signal mapping**: Maps input signals to variable names for each submodule
- **Memory state variables**: Names and initial values of latent state variables
- **Logit computation**: (Optional) Specifies which memory state variables are used to compute output logits

### Differentiable SINDy (`spice/resources/sindy_differentiable.py`)
- PyTorch-based implementation of SINDy for end-to-end gradient flow
- Enables joint optimization of RNN parameters and SINDy coefficients
- Supports custom candidate function libraries

### RNN Architecture (`spice/resources/rnn.py`)
- PyTorch-based neural network with disentangled submodules
- Each submodule learns a specific cognitive mechanism
- Supports participant embeddings for individual differences

### SPICE Training (`spice/resources/spice_training.py`)
- Training procedures combining RNN loss with SINDy regularization
- Implements the SINDy-SHRED-style joint optimization

**Objective Function:**
```
L_total = L_CE(y, ŷ) + λ_sindy * L_SINDy
```
- `L_CE`: Cross-entropy loss between predicted and actual actions (main behavioral prediction objective)
- `L_SINDy`: SINDy regularization loss pushing RNN dynamics toward sparse, interpretable equations
- `λ_sindy`: Regularization strength balancing predictive accuracy vs. interpretability

### Precoded Models (`spice/precoded/`)
- `RescorlaWagnerRNN`: Basic reinforcement learning with learning rates
- `choice.py`: Tracks chosen/unchosen action values separately
- `workingmemory.py`: Adds memory capacity for past rewards and choices

## SPICE Model Architecture

A SPICE model contains the following key components:

### `model.submodules_rnn`
Disentangled RNN submodules defined by their I/O specification:
- **Input size**: Manually specified, depends on input signals + optional participant embedding dimensions
- **Output size**: Always 1, mapping onto a single memory state variable
- **Optional**: Dropout for regularization
- Each submodule learns an independent cognitive mechanism (e.g., value updating, forgetting)

### `model.participant_embedding`
Similar to word embeddings in NLP:
- Each participant is represented as a learnable multidimensional vector
- Embeddings are fed to submodules to "parameterize" them according to participant characteristics
- Enables **hierarchical fitting** analogous to hierarchical Bayesian models
- Captures individual differences in cognitive parameters

### `model.sindy_coefficients`
Learnable coefficient dictionary for SINDy equations:
- Structure: `Dict[submodule_name, Tensor]` with one entry per submodule
- Each tensor has shape `(n_participants, n_candidate_terms)`
- Contains the weights for each candidate term per participant
- Enables participant-specific equation coefficients (individual differences in dynamics)
- Jointly optimized with RNN parameters during training

### `model.sindy_candidate_terms`
Library of basis functions for equation discovery:
- Structure: `Dict[submodule_name, ...]` with one entry per submodule
- Defines the functional form of possible equation terms (e.g., `1`, `x`, `x²`, `x*y`, `sin(x)`)
- Each submodule has its own candidate library based on its input signals
- Terms are evaluated on submodule inputs to construct the SINDy library matrix

### `model.sindy_coefficients_presence`
Binary mask indicating active (non-zero) coefficients:
- Structure: `Dict[submodule_name, Tensor]` matching `sindy_coefficients`
- Each tensor has shape `(n_participants, n_candidate_terms)`
- `1` = coefficient is active, `0` = coefficient has been thresholded to zero
- Updated during the sequential thresholding procedure

### `model.sindy_cutoff_patience_counters`
Patience counters for below-threshold coefficients:
- Structure: `Dict[submodule_name, Tensor]` matching `sindy_coefficients`
- Each tensor has shape `(n_participants, n_candidate_terms)`
- Tracks how many consecutive epochs each coefficient has remained below the threshold
- Prevents premature elimination of coefficients that temporarily dip below threshold
- Coefficient is only zeroed out when counter exceeds patience limit

### `model.sindy_degree_weights`
Penalty weights based on candidate term complexity/degree:
- Structure: `Dict[submodule_name, Tensor]` with one entry per submodule
- Each tensor has shape `(n_candidate_terms,)` — shared across participants
- Higher-degree polynomial terms receive stronger regularization penalties
- Encourages discovery of simpler, more interpretable equations
- Example: `x²` penalized more than `x`, `x³` penalized more than `x²`

### `model.use_sindy`
Boolean flag controlling SINDy integration:
- `True`: Use SINDy equations for forward pass (interpretable mode)
- `False`: Use RNN submodules for forward pass (neural network mode)
- Allows comparison between RNN predictions and SPICE (equation-based) predictions

## SINDy-Based Fitting Procedure

The SPICE training procedure extends standard SINDy with several enhancements for robust equation discovery:

### 1. Joint RNN-SINDy Optimization
```
L_total = L_CE(y, ŷ) + λ_sindy * L_SINDy
```
The RNN and SINDy coefficients are optimized jointly. The SINDy loss acts as a regularizer, pushing RNN dynamics toward a space where sparse equations can approximate them accurately.

### 2. Degree-Weighted Penalty
Higher-complexity candidate terms receive stronger L1/L2 penalties:
```
L_SINDy = Σᵢ degree_weight[i] * |coefficient[i]|
```
This encourages parsimony by favoring lower-degree terms (e.g., linear over quadratic), resulting in simpler, more interpretable equations.

### 3. Sequential Thresholding with Patience
Unlike standard STLSQ (Sequential Thresholded Least Squares), SPICE uses patience-based thresholding:

1. **Threshold check**: After each epoch, identify coefficients with magnitude below threshold `η`
2. **Patience counting**: Increment patience counter for below-threshold coefficients
3. **Elimination**: Only zero out coefficients when `patience_counter > patience_limit`
4. **Reset**: If coefficient rises above threshold, reset its patience counter to 0

This prevents premature elimination of coefficients that may temporarily fluctuate below the threshold during optimization, leading to more stable equation discovery.

### 4. Iterative Refinement
After thresholding, the remaining active coefficients are re-estimated via least squares on the reduced library, and the process repeats until convergence or maximum iterations.

## Development Guidelines

### Code Style
- Follow existing PyTorch conventions in the codebase
- Use type hints where present in existing code
- Maintain sklearn estimator API compatibility for `SpiceEstimator`
- Prefer descriptive variable names (e.g., `n_participants` over `n`)
- Use snake_case for functions/variables, PascalCase for classes

### File Organization
- Core framework code → `spice/`
- Research/paper code → `weinhardt2025/`
- New utility functions → `spice/utils/`
- New model architectures → `spice/precoded/`

### PyTorch Conventions
- Use `torch.nn.Module` for all neural network components
- Implement `forward()` method for all modules
- Use `torch.no_grad()` for inference
- Device handling: models should work on both CPU and CUDA

### Testing
- Test changes by running tests similar to `weinhardt2025/run.py`
- Verify sklearn interface compatibility (fit/predict pattern)
- Test both RNN and SPICE prediction paths

## Common Tasks

### Fitting a SPICE Model
```python
from spice.estimator import SpiceEstimator
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG

estimator = SpiceEstimator(
    rnn_class=RescorlaWagnerRNN,
    spice_config=RESCOLA_WAGNER_CONFIG,
    verbose=True,
)
estimator.fit(X, y)
features = estimator.spice_agent.get_spice_features()
```

### Running Paper Experiments
```bash
python weinhardt2025/run.py [args]
```

## Important Notes

- The `weinhardt2025/` directory contains research code for a paper in progress
- Core framework code lives in `spice/` - changes here affect the pip package
- SINDy regularization strength balances interpretability vs. predictive accuracy
- Individual differences are captured via participant embeddings

## Common Pitfalls

- **Shape mismatches**: Ensure `xs` has shape `(batch, timesteps, features)` and `ys` has shape `(batch, timesteps, actions)`
- **One-hot encoding**: Actions and rewards must be one-hot encoded, not integer indices
- **Device mismatch**: When using CUDA, ensure all tensors are on the same device
- **SINDy convergence**: If SINDy coefficients don't converge, try adjusting `λ_sindy` or increasing training epochs
- **Patience tuning**: Too low patience may prematurely eliminate important terms; too high delays sparsification
- **Degree weights**: Overly aggressive degree penalties may prevent discovery of necessary nonlinear terms
- **Participant IDs**: Must be included in features for individual difference modeling

## Documentation

Full documentation: https://whyhardt.github.io/SPICE/

## Architecture Decisions

- **Why sklearn interface?** Provides familiar API for researchers, enables pipeline integration
- **Why disentangled submodules?** Each cognitive mechanism can be independently interpretable
- **Why SINDy regularization during training?** Guides RNN toward dynamics that SINDy can approximate (SINDy-SHRED approach)
- **Why differentiable SINDy?** Enables end-to-end gradient flow for joint optimization
- **Why patience-based thresholding?** Prevents premature coefficient elimination during noisy optimization
- **Why degree-weighted penalties?** Encourages simpler equations (Occam's razor for equation discovery)