import torch
from spice import BaseRNN, SpiceConfig


class QLearning(BaseRNN):
    
    def __init__(self,
                 n_actions: int = 2,
                 n_participants: int = 1,
                 beta_reward: float = 3.0,
                 alpha_reward: float = 0.5,
                 alpha_penalty: float = 0.5,
                 forget_rate: float = 0.,
                 beta_choice: float = 0.,
                 fit_model: bool = False,
                 **kwargs,
        ):
        
        spice_config = SpiceConfig(
            library_setup={
                'value_reward_chosen': ['reward'],
                'value_reward_not_chosen': [],
                'value_choice': ['choice'],
            },
            memory_state=['value_reward', 'value_choice'],
        )
        
        super().__init__(
            spice_config=spice_config, 
            n_actions=n_actions, 
            n_participants=n_participants, 
            use_sindy=True, 
            sindy_polynomial_degree=2,
            sindy_ensemble_size=1)
                
        self.beta_reward = beta_reward
        self.alpha_reward = alpha_reward
        self.alpha_penalty = alpha_penalty
        self.forget_rate = forget_rate
        self.beta_choice = beta_choice
        
        # basic SPICE stuff
        self.rnn_training_finished = True  # rnn not necessary here
        self.setup_module(key_module='value_reward_chosen', input_size=1)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0)
        self.setup_module(key_module='value_choice', input_size=1)
        
        # specific to spice-based q-learning model
        self.coefficient_maps = {
            # asymmetric learning rate: 
            #   update = (alpha_penalty + (alpha_reward - alpha_penalty)*reward)*(reward-value_reward_chosen[t])
            #   update = -alpha_penalty value_reward_chosen[t] + alpha_reward reward + (alpha_reward-alpha_penalty) value_reward_chosen[t]*reward
            'value_reward_chosen': (
                ('value_reward_chosen', -self.alpha_penalty),
                ('reward', self.beta_reward*self.alpha_reward),
                ('value_reward_chosen*reward', self.alpha_penalty-self.alpha_reward),
                ),
            # forgetting:
            #   update = forget_rate*(self.state['value_reward']|_{t=0}-value_reward_not_chosen[t])
            #   update = forget_rate*Q_init + -forget_rate value_reward_not_chosen[t]
            'value_reward_not_chosen': (
                ('1', self.beta_reward*self.forget_rate*self.state['value_reward'][0, 0]),
                ('value_reward_not_chosen', -self.forget_rate),
            ),
            # choice perseverance:
            #   update = choice_perseverance choice
            'value_choice': (
                ('value_choice', -1),
                ('choice', self.beta_choice),
            ) 
        }
        
        # if the model is not about to be fitted but the given parameter values should be used then translate these values into the SINDy coefficients 
        if not fit_model:
            for module in self.get_modules():
                self.sindy_coefficients[module].requires_grad = False
                self.sindy_coefficients[module] *= 0
                for candidate_term, value in self.coefficient_maps[module]:
                    self.sindy_coefficients[module][:, 0, 0, self.sindy_candidate_terms[module].index(candidate_term)] = value

    def forward(self, inputs: torch.Tensor, prev_state = None, batch_first: bool = False):
        
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
        
        for timestep in spice_signals.timesteps:
            
            # perform reward update for chosen action
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                inputs=(
                    spice_signals.rewards[timestep],
                ),
                action_mask=spice_signals.actions[timestep],
                participant_index=spice_signals.participant_ids,
            )
            
            # perform forget update for unchosen action
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                participant_index=spice_signals.participant_ids,
            )
            
            # perform update for choice perseverance
            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                inputs=(
                    spice_signals.actions[timestep],
                ),
                participant_index=spice_signals.participant_ids,
            )
            
            spice_signals.logits[timestep] = self.state['value_reward'] + self.state['value_choice']
            
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.state
    