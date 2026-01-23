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
    
    
#     def new_sess(self, sample_parameters=False, **kwargs):
#     """Reset the agent for the beginning of a new session."""
    
#     super().new_sess()
    
#     # sample new parameters
#     if sample_parameters:
#       # sample scaling parameters (inverse noise temperatures)
#       self.betas['value_reward'], self.betas['value_choice'] = 0, 0
#       while self.betas['value_reward'] <= self.zero_threshold and self.betas['value_choice'] <=  self.zero_threshold:
        
#         self.betas['value_reward'] = np.random.beta(*self.compute_beta_dist_params(mean=0.5, var=self.parameter_variance['beta_reward']))
#         self.betas['value_choice'] = np.random.beta(*self.compute_beta_dist_params(mean=0.5, var=self.parameter_variance['beta_choice']))
#         # apply zero-threshold if applicable
#         self.betas['value_reward'] = self.betas['value_reward'] * 2 * self.mean_beta_reward if self.betas['value_reward'] > self.zero_threshold else 0
#         self.betas['value_choice'] = self.betas['value_choice'] * 2 * self.mean_beta_choice if self.betas['value_choice'] > self.zero_threshold else 0
      
#       # sample auxiliary parameters
#       self.forget_rate = np.random.beta(*self.compute_beta_dist_params(mean=self.mean_forget_rate, var=self.parameter_variance['forget_rate']))
#       self.forget_rate =  self.forget_rate * (self.forget_rate > self.zero_threshold)
      
#       self.alpha_choice = np.random.beta(*self.compute_beta_dist_params(mean=self.mean_alpha_choice, var=self.parameter_variance['alpha_choice']))
#       self.alpha_choice = self.alpha_choice * (self.alpha_choice > self.zero_threshold)
      
#       # sample learning rate; don't zero out; only check for applicability of asymmetric learning rates
#       self.alpha_reward = np.random.beta(*self.compute_beta_dist_params(mean=self.mean_alpha_reward, var=self.parameter_variance['alpha_reward']))
#       self.alpha_penalty = np.random.beta(*self.compute_beta_dist_params(mean=self.mean_alpha_penalty, var=self.parameter_variance['alpha_penalty']))
#       if np.abs(self.alpha_reward - self.alpha_penalty) < self.zero_threshold:
#         alpha_mean = np.mean((self.alpha_reward, self.alpha_penalty))
#         self.alpha_reward = alpha_mean
#         self.alpha_penalty = alpha_mean
        
#   def compute_beta_dist_params(self, mean, var):
#     n = mean * (1-mean) / var**2
#     a = mean * n
#     b = (1-mean) * n
#     return a, b

    # def check_parameter_variance(self, parameter_variance):
    #     if isinstance(parameter_variance, float):
    #       par_var_dict = {}
    #       for key in self.list_params:
    #         par_var_dict[key] = parameter_variance
    #       parameter_variance = par_var_dict
    #     elif isinstance(parameter_variance, dict):
    #       # check that all keys in parameter_variance are valid
    #       not_valid_keys = []
    #       for key in parameter_variance:
    #         if not key in self.list_params:
    #           not_valid_keys.append(key)
    #       if len(not_valid_keys) > 0:
    #         raise ValueError(f'Some keys in parameter_variance are not valid ({not_valid_keys}). Valid keys are {self.list_params}')
    #       # check that all parameters are available - set to 0 if a parameter is not available
    #       for key in self.list_params:
    #         if not key in parameter_variance:
    #           parameter_variance[key] = 0.
    #     return parameter_variance