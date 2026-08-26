# SPICE - Sparse and Interpretable Cognitive Equations

## Project Overview

SPICE is a framework for discovering symbolic cognitive mechanisms from behavioral data. It combines Recurrent Neural Networks (RNNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to extract interpretable mathematical equations that describe latent cognitive processes.

## Abstract

Discovering computational models that explain human cognition and behavior remains a central goal of cognitive science, yet the reliance on hand-crafted equations limits the range of cognitive mechanisms that can be uncovered. We introduce SPICE (Sparse and Interpretable Cognitive Equations), a framework that automates the discovery of mechanistically interpretable cognitive models directly from behavioral data. SPICE fits recurrent neural networks to capture latent cognitive dynamics and then applies sparse equation discovery to extract concise mathematical expressions describing those dynamics. Theory-guided priors make the approach data- and compute-efficient, while a hierarchical design reveals individual differences in the algorithmic structure of cognitive dynamics rather than in parameters alone. In simulations, SPICE accurately recovered the structure and parameters of known reinforcement learning models. Applied to human behavior in a two-armed bandit task, it uncovered new equations that outperformed existing models and revealed structural alterations in reinforcement learning mechanisms among participants with depression, such as a loss of nonlinear exploration dynamics regulating behavioral flexibility. This approach provides systematic insights into structural individual differences in cognitive mechanisms and establishes a foundation for automated discovery of interpretable behavioral models.

### Core Methodology

1. **RNN Training**: A task-specific RNN learns to predict human behavior, implicitly capturing latent cognitive mechanisms in disentangled submodules
2. **SINDy Regularization**: During training, SINDy equations act as regularizers, pushing submodule dynamics toward spaces amenable to SINDy candidate terms
3. **Equation Discovery**: SINDy approximates the fitted dynamics in each disentangled submodule, yielding interpretable symbolic equations, then a final refit stage recovers coefficients on the frozen RNN's dynamics

### The Concept Factorization

Per-participant coefficients are not free parameters. For each submodule the coefficient
matrix is factorized as

    A_pt = Z_pc · V_ct

where `V` is a **population-level dictionary of concepts** — sparse, unit-norm directions
over the candidate terms — and `Z` holds each participant's **non-negative loading** on
each concept. A term is therefore only ever interpretable together with the other terms
its concept owns, and a participant's structure is a *gate pattern over concepts* rather
than an arbitrary subset of terms.

Because SINDy targets the state *increment* (`h_next = h + dt * library @ coefficients`),
a Rescorla-Wagner update is a single concept: direction `(-1 Q, +1 r)` carrying one
loading `α` per participant, rather than two coefficients that happen to sum to one. More
generally a relaxation `Δx = λ(x* − x)` has direction encoding the *fixed point* and
loading encoding the *rate*.

Identifiability rests on four things, all enforced during training:

- **`Z ≥ 0`**, applied with the L1 as a proximal step (`z ← relu(z − lr·α)`) so the zeros
  are exact — an L1 loss term under Adam never produces one.
- **Unit-norm `V` rows**, re-fixed after every optimizer step. `Z @ V` is invariant under
  `(Z D, D⁻¹ V)`, so without this the cheapest way to shrink the penalty is to inflate `V`.
- **Undercomplete `C`** (defaults to `ceil(n_terms / 2)`); overcomplete dictionaries are exactly
  where identifiability fails. Concepts retire but never spawn, so `C` is a real hyperparameter.
- **The anchor condition**: each concept must own at least one term no other concept covers.
  Overlapping supports are allowed — a shared coefficient then becomes a *prediction*
  (`a_shared = z₁ + z₂`) rather than a free parameter — but a concept with no exclusive term
  can be mixed into the others without changing the fit. Enforced once after structure
  discovery converges, not during the search.

### What SPICE Models

Any trial-by-trial behavioral task where a participant (human or animal) makes repeated choices and the researcher wants to recover *interpretable* latent cognitive dynamics — not just predict the next action. Studies in `weinhardt2026/studies/` cover: reward-learning bandits with working memory, directed exploration, confidence-weighted learning under perceptual uncertainty, patch foraging, task-switching/cognitive-control effort, changepoint/volatility belief updating (linear and circular state spaces), collaborative visuospatial foraging, and primate social behavior. See [docs/studies.md](docs/studies.md) for what each study models and its population.

## Repository Structure

```
SPICE/
├── spice/                              # Core framework (backend / pip package)
│   ├── resources/
│   │   ├── estimator.py                # SpiceEstimator — scikit-learn compatible wrapper
│   │   ├── model.py                    # BaseModel — core RNN + SINDy architecture
│   │   ├── spice_utils.py              # SpiceConfig, SpiceDataset, SpiceSignals
│   │   ├── training/                   # Two-stage training pipeline (split by responsibility)
│   │   │   ├── fit.py                  # fit_spice orchestrator
│   │   │   ├── stage1.py               # Joint RNN + SINDy training against behaviour
│   │   │   ├── stage2.py               # SINDy refit on frozen trajectories (2.1 + 2.2)
│   │   │   ├── shooting.py             # Multi-step rollouts + post-step constraint projection
│   │   │   ├── trajectories.py         # Hidden-state trajectory collection/reshaping
│   │   │   ├── ridge.py                # Closed-form ridge initialization
│   │   │   ├── pruning.py              # Gate pruning + cross-ensemble consensus
│   │   │   ├── losses.py               # Loss functions and schedule helpers
│   │   │   └── reporting.py            # Terminal output
│   │   ├── sindy_differentiable.py     # Differentiable SINDy polynomial library
│   │   └── sindy_concept_init.py       # Data-driven seeding for the concept dictionary
│   ├── precoded/                       # Pre-built cognitive model architectures
│   │   ├── rescorlawagner.py           # Rescorla-Wagner learning model
│   │   ├── choice.py                   # Choice perseveration
│   │   ├── forgetting.py               # Forgetting mechanisms
│   │   ├── learningrate.py             # Dynamic learning rates
│   │   ├── interaction.py              # Interaction effects
│   │   ├── embedding.py                # Participant embeddings
│   │   ├── ddm.py                      # Drift Diffusion Model (within-trial dynamics)
│   │   ├── workingmemory.py            # Working memory with reward/choice buffers
│   │   └── workingmemory_*.py          # Working memory variants
│   └── utils/
│       ├── convert_dataset.py          # CSV ↔ SpiceDataset conversion pipeline
│       └── plotting.py                 # Visualization utilities
│
├── weinhardt2026/                      # Paper-specific code (fitting, benchmarking, analyses)
│   ├── run.py                          # Main entry point for training
│   ├── studies/                        # Self-contained study directories (see docs/studies.md)
│   ├── analysis/                       # Cross-study analysis pipelines (see docs/analyses.md)
│   └── utils/                          # Shared utilities (generative benchmarking, bandits, etc.)
│
├── docs/                                          # Documentation and tutorials
│   ├── training.md                                # Model/training internals (start here for how SPICE fits)
│   ├── analyses.md                                # Generative benchmarking + downstream analysis pipelines
│   ├── studies.md                                 # Per-study task/population summaries
│   └── guidelines_polynomial_amenable_architectures.md  # Architecture design guidelines
├── pyproject.toml                      # Package config (autospice v0.2.0, Python >=3.11)
├── setup.py                            # Installation
└── requirements.txt                    # Core dependencies
```

## Tech Stack

- **Language**: Python 3.11+
- **ML Framework**: PyTorch (2.7+)
- **API Style**: Scikit-learn estimator interface
- **Package Name**: `autospice` (pip installable)

## Commands

```bash
pip install autospice                    # Install from PyPI
pip install -e .                         # Install locally in editable mode
python weinhardt2026/run.py              # Fit SPICE model to dataset
```

---

## API at a Glance

Full details, current constructor signatures, and internals: **[docs/training.md](docs/training.md)**.

- **`SpiceConfig`** (`spice/resources/spice_utils.py`) — declares a model's architecture: which submodules exist (`library_setup`), the latent memory states they update (`memory_state`), and which states feed the output logits (`states_in_logit`).
- **`BaseModel`** (`spice/resources/model.py`) — the RNN + SINDy architecture task-specific models subclass. Register submodules with `setup_module()`, run them each trial with `call_module()`. Holds the concept factorization: `sindy_concept_directions` (`V`, shape `(C, T)`), `sindy_concept_loadings` (`Z`, shape `(E, P, X, C)`), plus the `sindy_concept_support` / `sindy_concept_gates` masks.
- **`SpiceDataset`** (`spice/resources/spice_utils.py`) — the training data container; build one from a behavioral CSV via `csv_to_dataset()` (`spice/utils/convert_dataset.py`).
- **`SpiceEstimator`** (`spice/resources/estimator.py`) — the sklearn-style entry point: `.fit(data, targets)`, `.predict(conditions)`, `.print_spice_model()`, `.get_concepts()`, `.get_concept_loadings()`, `.get_sindy_coefficients()`, `.count_spice_parameters()`, `.save_spice()`/`.load_spice()`.

**Precoded models** (`spice/precoded/`): Rescorla-Wagner, Choice Perseveration, Forgetting, Learning Rate, Interaction, Embedding, DDM, Working Memory (+ variants) — ready-made `BaseModel` subclasses for common cognitive mechanisms.

**Downstream analysis** (`weinhardt2026/analysis/`, see **[docs/analyses.md](docs/analyses.md)**): model evaluation (BIC/AIC/likelihood), model morphing (continuous structural trajectories along a behavioral axis), individual-differences regression on concept loadings, generative behavior comparison, behavioral clustering, and more — plus generative benchmarking (`weinhardt2026/utils/task.py`) for simulating new behavior from a fitted model.

---

## Coding Style

### Philosophy
- **Highly modular** over compact — prefer small, composable pieces
- **Slim over backward-compatible** — delete dead code rather than shimming it; no `_deprecated_*` wrappers or re-exports
- **General backend, specific frontends** — `BaseModel` and training infrastructure stay task-agnostic; all task-specific logic lives in `class MyModel(BaseModel)` subclasses
- **Keep docs in sync with the code, but not line-by-line** — don't rewrite documentation after every small edit. When a task, feature, or package is considered finished (a function/module is done, a parameter is removed, a pipeline stage changes behavior), ask the user whether `CLAUDE.md` / `docs/training.md` / `docs/analyses.md` / `docs/studies.md` should be updated to match, rather than silently leaving them stale or silently rewriting them unprompted.

### Conventions
- snake_case for functions/variables, PascalCase for classes
- Descriptive variable names (e.g., `n_participants` over `n`)
- Follow existing PyTorch conventions (`.forward()`, `torch.nn.Module`, device handling)
- Maintain sklearn estimator API compatibility for `SpiceEstimator`
- Type hints where present in existing code
- `@torch.compile` for performance-critical forward loops

### File Organization
- Core framework code → `spice/`
- Research/paper code → `weinhardt2026/`
- New utility functions → `spice/utils/`
- New model architectures → `spice/precoded/`

### Designing Polynomial-Amenable Architectures

When designing `BaseModel` subclasses, the architecture determines how well SINDy polynomials can approximate the learned RNN dynamics. Full guidelines (externalizing gating via action masks, keeping 1-3 control signals per module, matching polynomial degree to mechanism, etc.): **[docs/training.md](docs/training.md#designing-polynomial-amenable-architectures)** and [docs/guidelines_polynomial_amenable_architectures.md](docs/guidelines_polynomial_amenable_architectures.md).

## Common Pitfalls

A living list of gotchas (tensor device reassignment, action-mask overlap, `SpiceEstimator.predict()`'s return shape, `sindy_refit=False` behavior, etc.) lives in **[docs/training.md](docs/training.md#common-pitfalls)** — check there before debugging shape or gradient issues.

## Documentation

- **[docs/training.md](docs/training.md)** — `BaseModel`, `SpiceConfig`, `SpiceDataset`, `SpiceEstimator` internals; the two-stage training pipeline; dimension conventions; the precoded-model pattern; architecture design guidelines; common pitfalls.
- **[docs/analyses.md](docs/analyses.md)** — generative benchmarking and the full `weinhardt2026/analysis/` pipeline (model evaluation, morphing, concept distributions/individuals, behavioral clustering, reward-history kernels, parameter recovery), with a suggested analysis sequence for a new study.
- **[docs/studies.md](docs/studies.md)** — what each study in `weinhardt2026/studies/` models, its population, and its benchmark comparison model.
- Full hosted documentation: https://whyhardt.github.io/SPICE/
