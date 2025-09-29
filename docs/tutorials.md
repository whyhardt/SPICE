---
layout: default
title: Tutorials
nav_order: 5
has_children: true
---

# Tutorials

Welcome to the SPICE tutorials section. These tutorials will guide you through various aspects of using SPICE for cognitive modeling.

## Available Tutorials

1. [Data Preparation in SPICE](tutorials/0_data_preparation.html)
   - Learn how to prepare and format your data for SPICE
   - Understand the DatasetRNN class and data utilities
   - Overview dataset splitting and preprocessing techniques

2. [Basic Rescorla-Wagner Model](tutorials/1_rescorla_wagner.html)
   - Introduction to SPICE using a simple Rescorla-Wagner learning model
   - Learn how to set up and train your first SPICE model
   - Understand the basics of combining RNNs with equation discovery

3. [Rescorla-Wagner with Forgetting](tutorials/2_rescorla_wagner_forgetting.html)
   - Extend the basic model with forgetting mechanisms
   - Learn how to work with multiple cognitive mechanisms
   - Understand how SPICE discovers interaction effects

4. [Working with Hardcoded Equations](tutorials/3_hardcoded_equations.html)
   - Learn how to use predefined equations in SPICE
   - Understand when and why to use hardcoded equations
   - Compare performance with discovered equations

5. [Modeling Individual Differences](tutorials/4_individual_differences.html)
   - Learn how to capture individual differences in cognitive models
   - Work with participant-specific parameters
   - Analyze and interpret individual variations

6. [Choice Perseverance](tutorials/5_choice_perseverance.html)
   - Extend the basic model with choice perseverance mechanisms
   - Handle multiple dynamical variables per choice item

## Running the Tutorials

Each tutorial is available in two formats:
1. As an interactive Jupyter notebook in the `tutorials/` directory of the repository
2. As a web page in this documentation

To run the interactive notebooks:

1. Clone the SPICE repository:
   ```bash
   git clone https://github.com/whyhardt/SPICE.git
   cd SPICE
   ```

2. Install SPICE and its dependencies:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook tutorials/
   ```

## Prerequisites

- Basic understanding of Python programming
- Familiarity with machine learning concepts
- Basic knowledge of cognitive modeling principles

## Getting Help

If you encounter any issues while following the tutorials:
1. Check the [API Reference](../api.html) for detailed documentation
2. Visit our [GitHub repository](https://github.com/whyhardt/SPICE)
3. Open an issue on GitHub 