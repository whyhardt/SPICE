---
layout: default
title: Adding a Choice Perseverance Mechanism
parent: Tutorials
nav_order: 6
---

# Adding a Choice Perseverance Mechanism

In this tutorial, you will learn how to extend the model by adding a choice perseverance mechanism that captures non-goal-directed behavior.
In real-world decision-making, humans don't always choose based purely on expected rewards. Sometimes we tend to repeat previous choices simply because we made them before, independent of their actual value. The choice perseverance mechanism models this by:

Tracking which actions were chosen previously
Gradually increasing the preference for recently chosen actions
Gradually decreasing the preference for non-chosen actions
Allowing for dynamic adjustment of perseverance strength

This module incorporates an additional RNN module which dynamically handles choice-based preferences separately from reward-based learning.
Prerequisites
Before starting this tutorial, make sure you have:

## Prerequisites

Before starting this tutorial, make sure you have:
- Completed the [Basic Rescorla-Wagner Tutorial](2_rescorla_wagner.html)
- SPICE installed with all dependencies
- Understanding of basic reinforcement learning concepts

## 1. Data generation

First of all we have to generate a dataset with multiple participants. Let's start with two different ones.

We are going generate half the dataset with participant #1 and the other half with participant #2.


```python
# Uncomment the code below and execute the cell if you are using Google Colab

#!pip uninstall -y numpy pandas
#!pip install numpy==1.26.4 pandas==2.2.2
```


```python
# Uncomment the code below and execute the cell if you are using Google Colab

#!pip install autospice
```


```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
```


```python
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset
from spice.resources.rnn_utils import DatasetRNN

# Set up the environment
n_actions = 2
sigma = 0.2

environment = BanditsDrift(sigma=sigma, n_actions=n_actions)

# Set up the agent
agent = AgentQ(
    n_actions=n_actions,
    alpha_reward=0.3,
    forget_rate=0.2,
    beta_choice=1.,
    alpha_choice=1.,
)

# Create the dataset
n_trials = 100
n_sessions = 100

dataset, _, _ = create_dataset(
    agent=agent,
    environment=environment,
    n_trials=n_trials,
    n_sessions=n_sessions,
)

# set all participant ids to 0 since this dataset was generated only by one parameterization
dataset.xs[..., -1] = 0
```

## 2. Using the precoded model

First we setup and train the precoded SPICE model and inspect its behavior, before implementing it ourselves.


```python
from spice.estimator import SpiceEstimator
from spice.precoded import ChoiceRNN, CHOICE_CONFIG

spice_estimator = SpiceEstimator(
    rnn_class=ChoiceRNN,
    spice_config=CHOICE_CONFIG,
    learning_rate=1e-2,
    epochs=1024,
    n_participants=1,
)

spice_estimator.fit(dataset.xs, dataset.ys)

spice_estimator.print_spice_model()
```


```python
from spice.utils.plotting import plot_session

# Let's see how well the dynamics were fitted
agents = {'groundtruth': agent, 'rnn': spice_estimator.rnn_agent, 'spice': spice_estimator.spice_agent}
fig, axs = plot_session(agents, dataset.xs[0], signals_to_plot=['x_value_reward', 'x_value_choice'])
```

## 3. Implementing the RNN from Scratch

Below is the actual implementation of the RNN model. This RNN includes an additional group of reward-agnostic RNN-modules. These modules update choice-based values of the chosen and non-chosen options.

The structure of this RNN is shown in the following figure:

![](../figures/spice_rnn_choice.png)


```python
from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN

custom_config = SpiceConfig(
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
    control_parameters=['c_action', 'c_reward'],
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup = {
        'x_value_reward_chosen': ['c_reward'],
        'x_value_reward_not_chosen': [],
        'x_value_choice_chosen': [],
        'x_value_choice_not_chosen': [],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup = {
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }    
)

class CustomRNN(BaseRNN):

    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
        }

    def __init__(
        self,
        n_actions,
        n_participants,
        **kwargs,
    ):
        
        super(CustomRNN, self).__init__(n_actions=n_actions, embedding_size=8)
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=self.embedding_size)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = self.setup_constant(embedding_size=self.embedding_size)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size=self.embedding_size)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = inputs
        participant_id, _ = ids
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
            self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])
            
            # updates for x_value_reward
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(reward),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                )
            
            # updates for x_value_choice
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # updating the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
            
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice'] * self.betas['x_value_choice'](participant_embedding)
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
```
