---
layout: default
title: Discovering the Rescorla-Wagner Model
parent: Tutorials
nav_order: 2
---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/whyhardt/SPICE/blob/main/tutorials/1_rescorla_wagner.ipynb)

# Discovering the Rescorla-Wagner Model

This tutorial teaches how to use SPICE to discover the Rescorla-Wagner model directly from data.

First, we will train the precoded RNN already setup for the Rescorla-Wagner case. This will allow us to see the training process, its inputs and outputs and performance plots. Then, you will learn how to implement this model as a custom RNN.

This tutorial introduces SPICE using a simple Rescorla-Wagner learning model. You'll learn how to:
- Use the precoded Rescorla-Wagner RNN
- Train it on simulated data
- Extract and interpret the discovered equations
- Implement the Rescorla-Wagner model as a custom RNN

## Prerequisites

Before starting this tutorial, make sure you have:
- SPICE installed (`pip install autospice`)
- Basic understanding of reinforcement learning
- Familiarity with Python

## The Rescorla-Wagner Model

The Rescorla-Wagner model is a fundamental model of associative learning that describes how associations between stimuli and outcomes are learned through experience. The basic equation is:

ΔV = α(λ - V)

where:
- V is the associative strength
- α is the learning rate
- λ is the maximum possible associative strength
- ΔV is the change in associative strength

## 1. Data generation
First, we simulate a synthetic dataset from a Q-learning agent performing the two-armed bandit task.

In such bandit tasks the participant has to choose between several options across many trials and receives a reward $r$ each time after selecting one of them.

This reward is based on a reward probability $p(r)$.

In some experiments the reward probabilities of the different options are fixed and in others they have a more dynamic nature.

In our case, the reward probabilities are going to change trial-by-trial randomly based on a drift rate $\sigma$ according to

$p(r;t+1) \leftarrow p(r;t) + d$ with $d \sim \mathcal{N}(0, \sigma)$,

where $d$ is the current drift.

Let's set up the environment first.


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




    <torch._C.Generator at 0x1123c1690>




```python
from spice.resources.bandits import BanditsDrift
```


```python
n_actions = 2
sigma = 0.2

environment = BanditsDrift(sigma=sigma, n_actions=n_actions)
```

Let's see how the reward probabilities of the arms change across trials


```python
import numpy as np
import matplotlib.pyplot as plt

n_trials = 100
reward_probabilities = np.zeros((n_trials, n_actions))

for index_trials in range(n_trials):
    reward_probabilities[index_trials] = environment.reward_probs
    environment.step(choice=0)

for index_action in range(n_actions):
    plt.plot(reward_probabilities[..., index_action], label=f'Option {index_action+1}')
plt.legend()
plt.xlabel(r'Trial $t$')
plt.ylabel(r'$p(r)$')
plt.show()
```


    
![png](output_8_0.png)
    


Great! After setting up the environment, we can now go on to set up our participant!

The agent's behavior is defined by its parameters. These parameters are set once in the beginning but you can also draw new parameters from a distribution for each new session (i.e. performing $t$ trials).

Let's begin with the simpler case first and keep the parameters fixed for all trials.

We are going to set up a simple Rescorla-Wagner model which has only a learning rate $\alpha$ and an inverse noise temperature $\beta_{reward}$ and generate a dataset with it.


```python
from spice.resources.bandits import AgentQ, create_dataset

agent = AgentQ(
    n_actions=n_actions,
    alpha_reward=0.3,
)

dataset, _, _ = create_dataset(agent=agent, environment=environment, n_trials=100, n_sessions=100)

# set all participant ids to 0 since this dataset was generated only by one parameterization
dataset.xs[..., -1] = 0
```

    Creating dataset...


    100%|██████████| 100/100 [00:00<00:00, 236.53it/s]


Let the agent perform now the task and track how the agent's internal believes change across trials!


```python
from spice.utils.plotting import plot_session

fig, axs = plot_session(agents = {'groundtruth': agent}, experiment=dataset.xs[0])
plt.show()
```


    
![png](output_12_0.png)
    


The green and red ticks at the bottom mark whenever option 1 was chosen and rewarded/not rewarded.

You can take a look at the experiment results either in dataset (which is used for training the RNN) or in the experiments (which is a list of performed sessions; more human-readable)



```python
print('Inputs (choice item 1 selected; choice item 2 selected; reward 1; reward 2; session id):')
print(dataset.xs)

print('Targets (next choice):')
print(dataset.ys)
```

    Inputs (choice item 1 selected; choice item 2 selected; reward 1; reward 2; session id):
    tensor([[[ 0.,  1., -1.,  1.,  0.],
             [ 0.,  1., -1.,  0.,  0.],
             [ 0.,  1., -1.,  1.,  0.],
             ...,
             [ 1.,  0.,  1., -1.,  0.],
             [ 1.,  0.,  1., -1.,  0.],
             [ 1.,  0.,  0., -1.,  0.]],
    
            [[ 1.,  0.,  1., -1.,  0.],
             [ 0.,  1., -1.,  0.,  0.],
             [ 0.,  1., -1.,  1.,  0.],
             ...,
             [ 1.,  0.,  0., -1.,  0.],
             [ 0.,  1., -1.,  1.,  0.],
             [ 0.,  1., -1.,  1.,  0.]],
    
            [[ 1.,  0.,  0., -1.,  0.],
             [ 0.,  1., -1.,  0.,  0.],
             [ 1.,  0.,  0., -1.,  0.],
             ...,
             [ 0.,  1., -1.,  1.,  0.],
             [ 0.,  1., -1.,  1.,  0.],
             [ 0.,  1., -1.,  1.,  0.]],
    
            ...,
    
            [[ 1.,  0.,  1., -1.,  0.],
             [ 1.,  0.,  0., -1.,  0.],
             [ 1.,  0.,  1., -1.,  0.],
             ...,
             [ 1.,  0.,  1., -1.,  0.],
             [ 1.,  0.,  1., -1.,  0.],
             [ 1.,  0.,  1., -1.,  0.]],
    
            [[ 0.,  1., -1.,  0.,  0.],
             [ 1.,  0.,  0., -1.,  0.],
             [ 1.,  0.,  1., -1.,  0.],
             ...,
             [ 0.,  1., -1.,  1.,  0.],
             [ 0.,  1., -1.,  1.,  0.],
             [ 1.,  0.,  0., -1.,  0.]],
    
            [[ 0.,  1., -1.,  0.,  0.],
             [ 1.,  0.,  1., -1.,  0.],
             [ 1.,  0.,  0., -1.,  0.],
             ...,
             [ 1.,  0.,  1., -1.,  0.],
             [ 0.,  1., -1.,  0.,  0.],
             [ 1.,  0.,  0., -1.,  0.]]])
    Targets (next choice):
    tensor([[[0., 1.],
             [0., 1.],
             [0., 1.],
             ...,
             [1., 0.],
             [1., 0.],
             [1., 0.]],
    
            [[0., 1.],
             [0., 1.],
             [1., 0.],
             ...,
             [0., 1.],
             [0., 1.],
             [0., 1.]],
    
            [[0., 1.],
             [1., 0.],
             [0., 1.],
             ...,
             [0., 1.],
             [0., 1.],
             [0., 1.]],
    
            ...,
    
            [[1., 0.],
             [1., 0.],
             [1., 0.],
             ...,
             [1., 0.],
             [1., 0.],
             [1., 0.]],
    
            [[1., 0.],
             [1., 0.],
             [1., 0.],
             ...,
             [0., 1.],
             [1., 0.],
             [1., 0.]],
    
            [[1., 0.],
             [1., 0.],
             [1., 0.],
             ...,
             [0., 1.],
             [1., 0.],
             [1., 0.]]])


Now that we have our data, we can proceed to setup our RNN and train it!

## 2. Using the precoded Rescorla-Wagner RNN

First we will use the precoded RNN for discovering models like the Rescorla-Wagner model directly from data. We use SpiceEstimator to fit this RNN to the data we simulated above. The RNN modules and SPICE configuration are predetermined for discovering the Rescorla-Wagner model.

If `spice_participant_id` is not specified (set to `None`) then the SPICE model is fit to all participants. However, the fitting procedure would take a while. Instead, we set `spice_participant_id=0` to extract the SPICE model for participant 0.


```python
from spice.estimator import SpiceEstimator
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG

spice_estimator = SpiceEstimator(
    rnn_class=RescorlaWagnerRNN,
    spice_config=RESCOLA_WAGNER_CONFIG,
    learning_rate=1e-2,
    epochs=1024,
    verbose=False,
)

spice_estimator.fit(dataset.xs, dataset.ys)

spice_estimator.print_spice_model()
```

    
    Training the RNN...
    Epoch 1024/1024 --- L(Train): 0.5436556; Time: 0.03s; Convergence: 5.87e-05
    Maximum number of training epochs reached.
    Model did not converge yet.

    100%|██████████| 1/1 [00:00<00:00,  3.60it/s]

    SPICE modules:
    (x_value_reward_chosen)[k+1] = -0.411 1 + 0.682 x_value_reward_chosen[k] + 0.887 c_reward[k]


    


You can see the resulting equation in one of the last output lines.

It should be similar to `(x_value_reward_chosen)[k+1] = -0.411 1 + 0.682 x_value_reward_chosen[k] + 0.887 c_reward[k]`.

It's maybe not reminding you directly of the classic Rescorla-Wagner model 

`(x_value_reward_chosen)[k+1] = ...`

`... = (x_value_reward_chosen)[k] + alpha_reward (c_reward[k] - (x_value_reward_chosen)[k])`

`... = (1 - alpha_reward) (x_value_reward_chosen)[k] + alpha_reward c_reward[k]`,

which is implemented by the synthetic participant, but let's break the identified equation down.

1. The constant `-0.411 1` is applied equally to both arms without considering the any reward or current value. Therefore, it could also be left out (there's actually a way for doing that in the method `create_dataset_sindy` which we could utilize; later more about that).

2. The term `0.682 x_value_reward_chosen[k]` is actually pretty close to the classic model with `alpha_reward = 0.3`

3. The term `0.887 c_reward[k]` can be a bit irritating but makes total sense when you consider that the classic model has the scaling factor $\beta$ which is the inverse noise temperature with `beta_reward = 3`. Therefore, considering this scaling factor we get the parameter `0.887 c_reward[k] / beta_reward` $\approx$ `0.3 c_reward[k]`.

By interpreting the identified equation, we can see that the pipeline was able to fit the exact mechanism as implemented in the synthetic participant!

Let's see how our model behaves with respect to our synthetic participant.

In the following plot you can compare the action probabilities P(action), the Q-Value, the reward-based as well as the choice-based values.
(You can ignore for now the learning rate $\alpha$)


```python
from spice.utils.plotting import plot_session

# get analysis plot
agents = {'groundtruth': agent, 'rnn': spice_estimator.rnn_agent, 'spice': spice_estimator.spice_agent}

fig, axs = plot_session(agents, dataset.xs[0])
plt.show()
```


    
![png](output_21_0.png)
    


## 3. Implementing the RNN as a custom module

In this section we are going to implement the precoded RNN from scratch.

This RNN will inherit from the `BaseRNN`-class which itself inherits from `pytorch.nn.Module`. This is the base class for neural networks in the `PyTorch` framework.

Therefore the RNN has to implement a `forward`-method which is used for prediction. Further, it needs submodules to perform computations. These submodules are stored in the dictionary ` submodules_rnn` with the key `x_ModuleName`. The start of the key `x_` means that we are talking here about a memory state variable of the RNN.

Here, we are going to implement the simplest version of such a RNN. This RNN will update only the value of the chosen option based on the reward and leaves the values of the not chosen options untouched.

The structure of this RNN is shown in the following figure:

![](../figures/spice_rnn_rescorla_wagner.png)


```python
from spice.resources.rnn import BaseRNN

class CustomRNN(BaseRNN):
    
    # set up a dictionary with initial values for each state in memory
    init_values = {
        'x_value_reward': 0.5,
    }
    
    def __init__(
        self,
        n_actions,
        **kwargs,
    ):   
        super(CustomRNN, self).__init__(n_actions=n_actions)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1)
        
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
            
            # Let's perform the belief update for the reward-based value of the chosen option
            # since all values are given to the rnn-module (independent of each other), the chosen value is selected by setting the action to the chosen one
            # if we would like to perform a similar update by calling a rnn-module for the non-chosen action, we would set the parameter to action=1-action.
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=reward,
                )

            # and keep the value of the not-chosen option unchanged
            next_value_reward_not_chosen = self.state['x_value_reward'] * (1-action)
            
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen  # memory state = (0.8, 0.3) <- next_value = (0.8, 0) + (0, 0.3)
            
            # Now keep track of this value in the output array
            logits[timestep] = self.state['x_value_reward']
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        # self.state['x_value_reward'] = value_reward
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
```

Now that we implemented our RNN, we can train it to see how well it fits the behavior of our synthetic participant.


```python
from spice.estimator import SpiceConfig

custom_config = SpiceConfig(
    # A list of all names of the RNN-modules which are computing the RNN's memory state
    rnn_modules=['x_value_reward_chosen'],

    # A list of all the control signals which are used as inputs to any of the RNN-modules
    control_parameters=['c_action', 'c_reward'],

    # Setup of the SINDy library
    # Determines which terms are allowed as control inputs to each SINDy model in a dictionary.
    # The key is the SINDy-model name (same as RNN-module), value is a list of allowed control inputs from the list of control signals 
    library_setup={
        'x_value_reward_chosen': ['c_reward'],
    },

    # Setup of the filtering condition
    # Determines the filtering condition on which samples are selected as training samples for each SINDy-model.
    # Example:
    # Since each RNN-module processes all values at once (but independet from each other), we have to filter for the updates of interest.
    # In the case of the reward-based value of the chosen option this means to use only the chosen items and not the non-chosen ones. 
    # Therefore, we can set a filter condition to get rid of all value updates for non-chosen options.  
    # The filter dictionary has the following structure:
    # key -> the SINDy model name
    # value -> triplet of values:
    #   1. str: feature name to be used as a filter
    #   2. numeric: the numeric filter condition
    #   3. bool: remove feature from control inputs if not needed as input to the module
    # Multiple conditions can also be given as a list of triplets, e.g. [['c_action', 1, True], ['c_reward', 0, False]]
    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
    },
)
```

    Library setup is valid. All keys and features appear in the provided list of features.


You can now use the `SpiceEstimator` to fit the RNN to the data we generated above.


```python
from spice.estimator import SpiceEstimator


custom_spice_estimator = SpiceEstimator(
    rnn_class=CustomRNN,
    spice_config=custom_config,
    learning_rate=1e-2,
    epochs=1024,
)

custom_spice_estimator.fit(dataset.xs, dataset.ys)
custom_spice_estimator.print_spice_model()
```

## Saving and loading models

Below is how you can save and load SPICE models. You can specify `path_rnn`, `path_spice` or both to save or load either or both of them.


```python
# Save trained model to file
spice_estimator.save_spice(path_rnn='rnn_model.pkl', path_spice='spice_model.pkl')

# Load saved model
loaded_spice = SpiceEstimator(
    rnn_class=CustomRNN,
    spice_config=custom_config,
)

loaded_spice.load_spice(path_rnn='rnn_model.pkl', path_spice='spice_model.pkl')
```

## Next Steps

After completing this tutorial, you can:
1. Experiment with different parameter values
2. Try more complex environments
3. Move on to the [Rescorla-Wagner with Forgetting](2_rescorla_wagner_forgetting.html) tutorial

## Common Issues and Solutions

- **Poor Convergence**: Try increasing the number of epochs or adjusting the learning rate
- **Overfitting**: Reduce the hidden size or increase the dataset size
- **Unstable Training**: Adjust the optimizer parameters or reduce the learning rate
