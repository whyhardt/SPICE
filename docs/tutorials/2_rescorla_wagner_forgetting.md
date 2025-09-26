---
layout: default
title: Adding a Forgetting Mechanism
parent: Tutorials
nav_order: 3
---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/whyhardt/SPICE/blob/main/tutorials/2_rescorla_wagner_forgetting.ipynb)

# Adding a Forgetting Mechanism

In this tutorial, you will learn how to extend the basic Rescorla-Wagner model by adding a forgetting mechanism for the not-chosen actions.

In real-world learning scenarios, humans tend to forget information about options they haven't chosen recently. The forgetting mechanism models this by:
- Gradually decreasing the value of non-chosen actions
- Maintaining separate learning rates for chosen and non-chosen actions
- Allowing for dynamic adjustment of forgetting rates

This module incorporates an additional RNN module which dynamically handles the reward-based values for the not-chosen action.

## Prerequisites

Before starting this tutorial, make sure you have:
- Completed the [Basic Rescorla-Wagner Tutorial](2_rescorla_wagner.html)
- SPICE installed with all dependencies
- Understanding of basic reinforcement learning concepts

## 1. Data generation
First, we simulate a synthetic dataset from a Q-learning agent performing the two-armed bandit task, as in the previous tutorial. This time we specify a forgetting rate.


```python
# Uncomment the below code for Google Colab

#!pip uninstall -y numpy pandas
#!pip install numpy==1.26.4 pandas==2.2.2
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

# Set up the environment
n_actions = 2
sigma = 0.2

environment = BanditsDrift(sigma=sigma, n_actions=n_actions)

# Set up the agent
agent = AgentQ(
    n_actions=n_actions,
    alpha_reward=0.3,
    forget_rate=0.2,  # Here we set the forgetting rate to 0.2
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

## 2. Using the precoded Rescorla-Wagner RNN with Forgetting Mechanism

First we will use the precoded RNN with the forgetting mechanism. This is basically the Rescorla-Wagner RNN with an additional mechanism for the not chosen action. Later, you will see how to implement it yourself.


```python
from spice.estimator import SpiceEstimator
from spice.precoded import ForgettingRNN, FORGETTING_RNN_CONFIG

spice_estimator = SpiceEstimator(
    rnn_class=ForgettingRNN,
    spice_config=FORGETTING_RNN_CONFIG,
    learning_rate=1e-2,
    epochs=1024,
)

spice_estimator.fit(dataset.xs, dataset.ys)

spice_estimator.print_spice_model()
```

    
    Training the RNN...
    Epoch 1024/1024 --- L(Train): 0.6055849; Time: 0.06s; Convergence: 4.44e-07
    Maximum number of training epochs reached.
    Model did not converge yet.

    100%|██████████| 1/1 [00:00<00:00,  2.65it/s]

    SPICE modules:
    (x_value_reward_chosen)[k+1] = -0.375 1 + 0.712 x_value_reward_chosen[k] + 0.862 c_reward[k]
    (x_value_reward_not_chosen)[k+1] = 0.800 x_value_reward_not_chosen[k]


    


Let's inspect the discovered cognitive features. As you can see, there is a value update for the chosen action, based on the current value (`x_value_reward_chosen`) and the current reward (`c_reward`), same as the previous simple Rescorla-Wagner model.

Additionally, there is now a second mechanism, `x_value_reward_not_chosen` which gets updated based on the value of the non-chosen action, with a weight of 0.8. This is very close to the original data, considering we set a forgetting rate of 0.2, which means the values will be retained with a weight of 0.8.

Now let's see how our model behaves with respect to our synthetic participant.

In the following plot you can compare the action probabilities P(action), the Q-Value, the reward-based as well as the choice-based values.
(You can ignore for now the learning rate $\alpha$)


```python
from spice.utils.plotting import plot_session
import matplotlib.pyplot as plt

# get analysis plot
agents = {'groundtruth': agent, 'rnn': spice_estimator.rnn_agent, 'spice': spice_estimator.spice_agent}

fig, axs = plot_session(agents, dataset.xs[0])
plt.show()
```


    
![png](output_11_0.png)
    


## 3. Implementing the RNN as a custom module

Now, we are going to implement the RNN that discovers RW model with a forgetting mechanism. This RNN will update the values of both the chosen option, as in the previous tutorial, and has an additional module for the not chosen option.

The structure of this RNN is shown in the following figure:

![](../figures/spice_rnn_forgetting.png)

As you can see we are going to add one module which will be responsible for updating the non-chosen value.


```python
from spice.resources.rnn import BaseRNN
from spice.estimator import SpiceConfig
    

FORGETTING_RNN_CONFIG = SpiceConfig(
    # Add already here the new module and update the library and filter setup.
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen'],
    
    control_parameters=['c_action', 'c_reward'],

    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup={
        'x_value_reward_chosen': ['c_reward'],
        'x_value_reward_not_chosen': [],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
    },
)

class ForgettingRNN(BaseRNN):
    
    init_values = {
        'x_value_reward': 0.5,
    }
    
    def __init__(
        self,
        n_actions,
        **kwargs,
    ):
        super(ForgettingRNN, self).__init__(n_actions=n_actions)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0)
        
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
                
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            
            # Let's perform the belief update for the reward-based value of the chosen option
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=reward,
                activation_rnn=torch.nn.functional.leaky_relu
                )

            # Now a RNN-module updates the not-chosen reward-based value instead of keeping it the same
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                )
            
            # keep track of the updated value in the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            
            # Now keep track of this value in the output array
            logits[timestep] = self.state['x_value_reward']
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
```

## Common Issues and Solutions

- **Unstable Learning**: Try reducing the learning rate or increasing batch size
- **Poor Forgetting**: Adjust the forgetting rate or increase training data
- **Convergence Issues**: Increase the number of epochs or adjust optimizer parameters

## Next Steps

After completing this tutorial, you can:
1. Experiment with different forgetting rates
2. Implement more complex forgetting mechanisms
3. Move on to [Working with Hardcoded Equations](3_hardcoded_equations.html)
