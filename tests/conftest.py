"""
Pytest configuration and fixtures for SPICE tests.
"""
import pytest
import numpy as np
import torch


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    n_participants = 3
    n_trials = 10
    n_features = 5
    n_actions = 2
    
    return {
        'n_participants': n_participants,
        'n_trials': n_trials,
        'n_actions': n_actions,
        'conditions': np.random.rand(n_participants, n_trials, n_features),
        'targets': np.random.randint(0, n_actions, size=(n_participants, n_trials, n_actions))
    }


@pytest.fixture
def model_config():
    """Basic model configuration for testing."""
    return {
        'hidden_size': 8,
        'n_actions': 2,
        'dropout': 0.25,
        'n_participants': 3,
        'n_experiments': 2,
        'rnn_modules': ['value', 'choice'],
        'control_parameters': ['reward', 'action'],
        'sindy_library_config': {
            'value': ['reward'],
            'choice': ['action']
        },
        'sindy_filter_config': {}
    } 