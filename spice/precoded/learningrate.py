from ..resources.estimator import SpiceConfig
from ..resources.model import BaseModel

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) DYNAMIC LEARNING RATE
# 2) HARD-CODED REWARD-PREDICTION-ERROR FOR CHOSEN ACTION
# 3) VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

def reward_prediction_error_update(value_reward, learning_rate, reward):
    """Delta rule: the new reward-based value of the chosen option."""
    return value_reward + learning_rate * (reward - value_reward)


CONFIG = SpiceConfig(
    discovered_modules={
        'learning_rate_reward': ['reward', 'value_reward'],
        'value_reward_not_chosen': [],
    },
    # hard-coded module: not approximated by SINDy, its functional form is assumed to be known
    fixed_modules={
        'value_reward_chosen': (reward_prediction_error_update, ['value_reward', 'learning_rate_reward', 'reward']),
    },
    # per-participant inverse noise temperature, positive by softplus
    constants={
        'inverse_temperature': torch.nn.functional.softplus,
    },
    memory_state={
            'value_reward': 0.5,
            'learning_rate_reward': 0,
    }
)


class SpiceModel(BaseModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # participant embedding: only used for the inverse temperature
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)

        # set up the submodules
        self.setup_module(key_module='learning_rate_reward', input_size=2, embedding_size=0)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0, embedding_size=0)
        
    def forward(self, inputs, prev_state=None):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        spice_signals = self.init_forward_pass(inputs, prev_state)

        # per-participant inverse noise temperature: computed once from the embedding
        inverse_temperature = self.call_module(
            key_module='inverse_temperature',
            participant_embedding=self.participant_embedding(spice_signals.participant_ids),
        )
                
        for timestep in spice_signals.trials:
            
            # Let's compute the learning rate dynamically (raw value; the module and the memory stay
            # free of activations so that SINDy can recover its equation)
            self.call_module(
                key_module='learning_rate_reward',
                key_state='learning_rate_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    spice_signals.feedback[timestep],
                    self.state['value_reward'],
                    ),
            )

            # The sigmoid keeps the learning rate in (0, 1), as required by the delta rule:
            # the hard-coded reward-prediction error cannot rescale the learning rate itself,
            # so the scale of the values is carried by the inverse noise temperature instead
            learning_rate_reward = torch.sigmoid(self.state['learning_rate_reward'])

            # Belief update for the reward-based value of the chosen option (hard-coded delta rule)
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    self.state['value_reward'],
                    learning_rate_reward,
                    spice_signals.feedback[timestep],
                    ),
                )

            # Update of the not-chosen reward-based value
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                )
            
            # logits scaled by the per-participant inverse noise temperature
            spice_signals.logits[timestep] = self.state['value_reward'] * inverse_temperature
            
        # post-process the forward pass
        spice_signals = self.post_forward_pass(spice_signals)
        
        return spice_signals.logits, self.get_state()
