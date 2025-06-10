---
layout: default
title: Installation
nav_order: 2
---

# Installation Guide

SPICE can be installed using pip or by building from source. Choose the method that best suits your needs.

## Quick Installation

The fastest way to get started with SPICE is to install it via pip:

```bash
pip install autospice
```

## Installing from Source

For the latest development version or if you want to contribute to SPICE, you can install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/whyhardt/SPICE.git
   cd SPICE
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

## Requirements

SPICE requires Python 3.8 or later. The main dependencies are:

- PyTorch
- NumPy
- SciPy
- Scikit-learn
- PySINDy

For a complete list of dependencies, see the `requirements.txt` file in the repository.

## Verifying Installation

To verify that SPICE is installed correctly, you can run:

```python
from spice.estimator import SpiceEstimator
print("SPICE installed successfully!")
```

## Development Installation

If you plan to contribute to SPICE, you should install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

This will install additional packages needed for testing and development.

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you have Python 3.8 or later installed
2. Verify that pip is up to date: `pip install --upgrade pip`
3. Check that all dependencies are properly installed
4. If problems persist, please open an issue on our [GitHub repository](https://github.com/whyhardt/SPICE/issues) 