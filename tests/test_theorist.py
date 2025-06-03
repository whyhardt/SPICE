"""
Unit tests for the RNN-SINDy theorist.
"""
import pytest
import numpy as np
import torch
from spice import rnn_sindy_theorist
from spice.resources import BaseRNN


class SimpleRNN(BaseRNN):
    """Simple RNN implementation for testing."""
    def __init__(self, n_actions, hidden_size, dropout, n_participants=0, n_experiments=0):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(
            input_size=5,  # Fixed for testing
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, n_actions)
        self.dropout = torch.nn.Dropout(dropout)
        
        if n_participants > 0:
            self.participant_embedding = torch.nn.Embedding(n_participants, 1)
        if n_experiments > 0:
            self.experiment_embedding = torch.nn.Embedding(n_experiments, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden


def test_theorist_initialization(model_config):
    """Test that the theorist can be properly initialized."""
    model = rnn_sindy_theorist(
        rnn_class=SimpleRNN,
        **model_config
    )
    assert model is not None
    assert isinstance(model.rnn_model, SimpleRNN)
    assert model.hidden_size == model_config['hidden_size']
    assert model.n_actions == model_config['n_actions']


def test_theorist_fit(model_config, sample_data):
    """Test the fit method of the theorist."""
    model = rnn_sindy_theorist(
        rnn_class=SimpleRNN,
        **model_config
    )
    
    model.fit(sample_data['conditions'], sample_data['targets'])
    
    # Check that the model components are properly initialized after fitting
    assert model.rnn_agent is not None
    assert model.sindy_agent is not None
    assert model.sindy_features is not None


def test_theorist_predict(model_config, sample_data):
    """Test the predict method of the theorist."""
    model = rnn_sindy_theorist(
        rnn_class=SimpleRNN,
        **model_config
    )
    
    model.fit(sample_data['conditions'], sample_data['targets'])
    pred_rnn, pred_sindy = model.predict(sample_data['conditions'])
    
    # Check shapes
    expected_shape = (
        sample_data['n_participants'],
        sample_data['n_trials'],
        sample_data['n_actions']
    )
    assert pred_rnn.shape == expected_shape
    assert pred_sindy.shape == expected_shape
    
    # Check that predictions are probabilities
    assert np.all(pred_rnn >= 0) and np.all(pred_rnn <= 1)
    assert np.all(pred_sindy >= 0) and np.all(pred_sindy <= 1)
    assert np.allclose(np.sum(pred_rnn, axis=-1), 1.0, atol=1e-6)
    assert np.allclose(np.sum(pred_sindy, axis=-1), 1.0, atol=1e-6)


def test_get_sindy_features(model_config, sample_data):
    """Test retrieving SINDy features."""
    model = rnn_sindy_theorist(
        rnn_class=SimpleRNN,
        **model_config
    )
    
    # Should raise error before fitting
    with pytest.raises(ValueError):
        model.get_sindy_features()
    
    model.fit(sample_data['conditions'], sample_data['targets'])
    features = model.get_sindy_features()
    
    # Check that features are returned for each participant
    assert isinstance(features, dict)
    assert len(features) == sample_data['n_participants']


def test_get_participant_embeddings(model_config, sample_data):
    """Test retrieving participant embeddings."""
    model = rnn_sindy_theorist(
        rnn_class=SimpleRNN,
        **model_config
    )
    
    model.fit(sample_data['conditions'], sample_data['targets'])
    embeddings = model.get_participant_embeddings()
    
    assert embeddings is not None
    assert len(embeddings) == sample_data['n_participants']
    assert all(isinstance(emb, torch.Tensor) for emb in embeddings.values())


def test_get_experiment_embeddings(model_config, sample_data):
    """Test retrieving experiment embeddings."""
    model = rnn_sindy_theorist(
        rnn_class=SimpleRNN,
        **model_config
    )
    
    model.fit(sample_data['conditions'], sample_data['targets'])
    embeddings = model.get_experiment_embeddings()
    
    assert embeddings is not None
    assert len(embeddings) == model_config['n_experiments']
    assert all(isinstance(emb, torch.Tensor) for emb in embeddings.values()) 