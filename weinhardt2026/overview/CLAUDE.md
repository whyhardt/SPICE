# SPICE - Sparse and Interpretable Cognitive Equations

## Project Overview

SPICE is a framework for discovering symbolic cognitive mechanisms from behavioral data. It combines Recurrent Neural Networks (RNNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to extract interpretable mathematical equations that describe latent cognitive processes.

## Abstract

Discovering computational models that explain human cognition and behavior remains a central goal of cognitive science, yet the reliance on hand-crafted equations limits the range of cognitive mechanisms that can be uncovered. We introduce SPICE (Sparse and Interpretable Cognitive Equations), a framework that automates the discovery of mechanistically interpretable cognitive models directly from behavioral data. SPICE fits recurrent neural networks to capture latent cognitive dynamics and then applies sparse equation discovery to extract concise mathematical expressions describing those dynamics. Theory-guided priors make the approach data- and compute-efficient, while a hierarchical design reveals individual differences in the algorithmic structure of cognitive dynamics rather than in parameters alone. In simulations, SPICE accurately recovered the structure and parameters of known reinforcement learning models. Applied to human behavior in a two-armed bandit task, it uncovered new equations that outperformed existing models and revealed structural alterations in reinforcement learning mechanisms in clinical groups (*placeholder: the specific group and mechanism are still to be determined from the stability runs*). This approach provides systematic insights into structural individual differences in cognitive mechanisms and establishes a foundation for automated discovery of interpretable behavioral models.

### Core Methodology

1. **RNN Training**: A task-specific RNN learns to predict human behavior, implicitly capturing latent cognitive mechanisms in disentangled submodules
2. **SINDy Regularization**: During training, SINDy equations act as regularizers, pushing submodule dynamics toward spaces amenable to SINDy candidate terms
3. **Equation Discovery**: SINDy approximates the fitted dynamics in each disentangled submodule, yielding interpretable symbolic equations, then a final refit stage recovers coefficients on the frozen RNN's dynamics

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
│   │   ├── spice_training.py           # Two-stage training pipeline
│   │   └── sindy_differentiable.py     # Differentiable SINDy polynomial library
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
│       ├── plotting.py                 # Visualization utilities
│       └── analysis/                   # Generic analyses of any directory of SPICE fits (no study names):
│                                       #   equation_discovery.py (mechanisms, correlations, beta effects),
│                                       #   coefficient_dendrogram.py, coefficient_betas.py, loading.py
│
├── weinhardt2026/                      # Paper-specific code (fitting, benchmarking, analyses)
│   ├── run.py                          # Main entry point for training
│   ├── studies/                        # Self-contained study directories (see docs/studies.md); per study: <study>.py (fit/load + quick evaluation),
│   │                                   #   spice_/benchmarking_/simulation_<study>.py, data/, params/ (final models), results/<item>/
│   ├── analysis/                       # Paper runners, each writing CSV + figures (see docs/analyses.md):
│   │                                   #   discovery/ (equation discovery for every study), evaluation/ (model evaluation, PPC),
│   │                                   #   validation/ (parameter recovery, equation stability, model_identification/),
│   │                                   #   grid_search/ (SPICE + benchmark/GRU; README.md = checkpoint names), explorative/ (morphing)
│   ├── results/                        # Paper results across studies (README.md = index of all paper results)
│   └── utils/                          # Shared utilities (study registry, final models, study simulation modules, generative benchmarking, ...)
│
├── docs/                                          # Documentation and tutorials
│   ├── training.md                                # Model/training internals (start here for how SPICE fits)
│   ├── analyses.md                                # Results layout + paper analysis pipelines
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
- **`BaseModel`** (`spice/resources/model.py`) — the RNN + SINDy architecture task-specific models subclass. Register submodules with `setup_module()`, run them each trial with `call_module()`.
- **`SpiceDataset`** (`spice/resources/spice_utils.py`) — the training data container; build one from a behavioral CSV via `csv_to_dataset()` (`spice/utils/convert_dataset.py`).
- **`SpiceEstimator`** (`spice/resources/estimator.py`) — the sklearn-style entry point: `.fit(data, targets)`, `.predict(conditions)`, `.print_spice_model()`, `.get_sindy_coefficients()`, `.save_spice()`/`.load_spice()`.

**Precoded models** (`spice/precoded/`): Rescorla-Wagner, Choice Perseveration, Forgetting, Learning Rate, Interaction, Embedding, DDM, Working Memory (+ variants) — ready-made `BaseModel` subclasses for common cognitive mechanisms.

**Generic analyses** (`spice/utils/analysis/`): equation discovery on any directory of fits of one SPICE model (`python -m spice.utils.analysis.equation_discovery --models-dir ... --module ... --data ...`) — mechanisms recurring across seeds (joint support), the reduced equation matrix, the correlations between mechanisms and their structural beta effects on a criterion.

**Paper analyses** (`weinhardt2026/analysis/`; see **[docs/analyses.md](docs/analyses.md)**), all on the final models (`studies/<study>/params/`, 10 seeds; `weinhardt2026/utils/final_models.py`): equation discovery per study (mechanisms, correlations, beta effects), model evaluation (held-out prediction, PPC, AIC/BIC), posterior predictive checks (simulation via `studies/<study>/simulation_<study>.py`), parameter recovery (precision/recall/F1 per dataset size and ground-truth model), equation stability across seeds, model identification (confusion matrix SPICE vs benchmark), the SPICE and benchmark/GRU grid searches and model morphing (dezfouli2019). Outputs: `weinhardt2026/results/<item>/` (across studies) and `weinhardt2026/studies/<study>/results/<item>/` (per study); index in `weinhardt2026/results/README.md`.

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

### Figures
- Every analysis (generic in `spice/utils/analysis/` or paper runner in `weinhardt2026/analysis/`) writes a CSV and one **annotated** figure (`<name>.pdf/.png`: cell values, significance marks, titles). A **clean** version (`<name>_clean.pdf/.png`: marks, axes, tick labels and colourbars only) is written when asked for (`clean=True` / `--clean`); the paper runners ask for it. Not needed for brainstorming or scratchpad figures.
- Figures and CSVs are written only to results folders: `weinhardt2026/results/<analysis>/` (across studies) or `weinhardt2026/studies/<study>/results/<analysis>/` (one study) — never into code folders.

### File Organization
- Core framework code → `spice/`
- Research/paper code → `weinhardt2026/`
- New utility functions → `spice/utils/`
- Generic, study-agnostic analyses → `spice/utils/analysis/`; paper runners → `weinhardt2026/analysis/<category>/`
- New model architectures → `spice/precoded/`

### Designing Polynomial-Amenable Architectures

When designing `BaseModel` subclasses, the architecture determines how well SINDy polynomials can approximate the learned RNN dynamics. Full guidelines (externalizing gating via action masks with an anchor per split pair, keeping the candidate library small, inputs in [-1, 1], additive logits, etc.): **[docs/training.md](docs/training.md#designing-polynomial-amenable-architectures)** and [docs/guidelines_polynomial_amenable_architectures.md](docs/guidelines_polynomial_amenable_architectures.md).

## Common Pitfalls

A living list of gotchas (tensor device reassignment, action-mask overlap, `SpiceEstimator.predict()`'s return shape, `sindy_refit=False` behavior, etc.) lives in **[docs/training.md](docs/training.md#common-pitfalls)** — check there before debugging shape or gradient issues.

## Documentation

- **[docs/training.md](docs/training.md)** — `BaseModel`, `SpiceConfig`, `SpiceDataset`, `SpiceEstimator` internals; the two-stage training pipeline; dimension conventions; the precoded-model pattern; architecture design guidelines; common pitfalls.
- **[docs/analyses.md](docs/analyses.md)** — the results layout, generative benchmarking and every paper analysis (model evaluation, parameter recovery, mechanisms, beta effects, equation stability, model identification, posterior predictive checks, equation tables, morphing, grid search), with a suggested analysis sequence for a new study.
- **[docs/studies.md](docs/studies.md)** — what each study in `weinhardt2026/studies/` models, its population, and its benchmark comparison model.
- Full hosted documentation: https://whyhardt.github.io/SPICE/
