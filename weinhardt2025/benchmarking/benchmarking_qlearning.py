from typing import Union
import torch

from spice import BaseRNN, SpiceConfig


class QLearning(BaseRNN):
    
    def __init__(self,
                 n_actions: int = 2,
                 n_participants: int = 1,
                 n_experiments: int = 1,
                 beta_reward: Union[float, torch.Tensor] = 1.0,
                 alpha_reward: Union[float, torch.Tensor] = 0.5,
                 alpha_penalty: Union[float, torch.Tensor] = 0.5,
                 forget_rate: Union[float, torch.Tensor] = 0.,
                 beta_choice: Union[float, torch.Tensor] = 0.,
                 alpha_choice: Union[float, torch.Tensor] = 1.,
                 counterfactual_learning: Union[float, torch.Tensor] = 0.,
                 fit_full_model: bool = False,
                 **kwargs,
        ):
        
        spice_config = SpiceConfig(
            library_setup={
                'value_reward_chosen': ['reward[t]'],
                'value_reward_not_chosen': [],#['reward_chosen_success', 'reward_chosen_fail'],
                'value_choice': ['choice[t]'],
            },
            memory_state=['value_reward', 'value_choice'],
        )
        
        super().__init__(
            spice_config=spice_config, 
            n_actions=n_actions, 
            n_participants=n_participants,
            n_experiments=n_experiments, 
            use_sindy=True, 
            sindy_polynomial_degree=2,
            sindy_ensemble_size=1)
        self.rnn_training_finished = True
        
        self.beta_reward = torch.full((self.n_participants, self.n_experiments), beta_reward) if isinstance(beta_reward, float) else beta_reward
        self.alpha_reward = torch.full((self.n_participants, self.n_experiments), alpha_reward) if isinstance(alpha_reward, float) else alpha_reward
        self.alpha_penalty = torch.full((self.n_participants, self.n_experiments), alpha_penalty) if isinstance(alpha_penalty, float) else alpha_penalty
        self.forget_rate = torch.full((self.n_participants, self.n_experiments), forget_rate) if isinstance(forget_rate, float) else forget_rate
        self.beta_choice = torch.full((self.n_participants, self.n_experiments), beta_choice) if isinstance(beta_choice, float) else beta_choice
        self.alpha_choice = torch.full((self.n_participants, self.n_experiments), alpha_choice) if isinstance(alpha_choice, float) else alpha_choice
        self.countefactual_learning = torch.full((self.n_participants, self.n_experiments), counterfactual_learning) if isinstance(counterfactual_learning, float) else counterfactual_learning
        
        # basic SPICE stuff
        self.rnn_training_finished = True  # rnn not necessary here
        self.setup_module(key_module='value_reward_chosen', input_size=1, include_bias=True)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0, include_bias=True)
        self.setup_module(key_module='value_choice', input_size=1, include_bias=True)
        
        if not fit_full_model:
            self.update_coefficients(
                parameters={
                    'beta_reward': self.beta_reward.clone(),
                    'alpha_reward': self.alpha_reward.clone(),
                    'alpha_penalty': self.alpha_penalty.clone(),
                    'forget_rate': self.forget_rate.clone(),
                    'beta_choice': self.beta_choice.clone(),
                    'alpha_choice': self.alpha_choice.clone(),
                    'countefactual_learning': self.countefactual_learning.clone(),
                },
                participant_id=torch.arange(0, self.n_participants),
                experiment_id=torch.arange(0, self.n_experiments),
            )

    def forward(self, inputs: torch.Tensor, prev_state = None, batch_first: bool = False):
        
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
        
        chosen_reward_success = spice_signals.rewards.sum(dim=-1, keepdim=True).repeat(1, 1, self.n_actions) == 1
        chosen_reward_fail = spice_signals.rewards.sum(dim=-1, keepdim=True).repeat(1, 1, self.n_actions) != 1
        
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
                inputs=(
                    # chosen_reward_success[timestep],
                    # chosen_reward_fail[timestep],
                ),
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

    def update_coefficients(self, parameters: dict, participant_id: Union[int, torch.Tensor] = None, experiment_id: Union[int, torch.Tensor] = None):
        if participant_id is None:
            participant_id = torch.arange(0, self.n_participants)
        
        if experiment_id is None:
            experiment_id = torch.arange(0, self.n_experiments)
        
        for parameter in parameters:
            if hasattr(self, parameter):
                model_parameters = getattr(self, parameter)
                model_parameters[participant_id.unsqueeze(1), experiment_id] = parameters[parameter]
                setattr(self, parameter, model_parameters)
            else:
                raise KeyError(f"The QLearning model has no attribute {parameter}.")
            
        # specific to spice-based q-learning model
        coefficient_maps = {
            # asymmetric learning rate: 
            #   update = (alpha_penalty + (alpha_reward - alpha_penalty)*reward)*(reward-value_reward_chosen[t])
            #   update = -alpha_penalty value_reward_chosen[t] + alpha_reward reward + (alpha_reward-alpha_penalty) value_reward_chosen[t]*reward
            'value_reward_chosen': (
                ('value_reward_chosen', -self.alpha_penalty[participant_id.unsqueeze(1), experiment_id] * (self.beta_reward[participant_id.unsqueeze(1), experiment_id] > 0) ),
                ('reward[t]', self.beta_reward[participant_id.unsqueeze(1), experiment_id]*self.alpha_reward[participant_id.unsqueeze(1), experiment_id]),
                ('value_reward_chosen*reward[t]', self.alpha_penalty[participant_id.unsqueeze(1), experiment_id]-self.alpha_reward[participant_id.unsqueeze(1), experiment_id] * (self.beta_reward[participant_id.unsqueeze(1), experiment_id] > 0) ),
                ),
            # forgetting:
            #   update = forget_rate*(self.state['value_reward']|_{t=0}-value_reward_not_chosen[t])
            #   update = forget_rate*Q_init + -forget_rate value_reward_not_chosen[t]
            'value_reward_not_chosen': (
                # ('1', self.beta_reward[participant_id.unsqueeze(1), experiment_id]*self.forget_rate[participant_id.unsqueeze(1), experiment_id]*self.state['value_reward'][0, 0]),
                ('value_reward_not_chosen', -self.forget_rate[participant_id.unsqueeze(1), experiment_id] * (self.beta_reward[participant_id.unsqueeze(1), experiment_id] > 0) ),
                # ('reward_chosen_success', -self.beta_reward[participant_id.unsqueeze(1), experiment_id]*self.countefactual_learning[participant_id.unsqueeze(1), experiment_id]),
                # ('reward_chosen_fail', self.beta_reward[participant_id.unsqueeze(1), experiment_id]*self.countefactual_learning[participant_id.unsqueeze(1), experiment_id]),
            ),
            # choice perseverance:
            #   update = choice_perseverance choice
            'value_choice': (
                ('value_choice', -self.alpha_choice[participant_id.unsqueeze(1), experiment_id] * (self.beta_choice[participant_id.unsqueeze(1), experiment_id] > 0) ),
                ('choice[t]', self.beta_choice[participant_id.unsqueeze(1), experiment_id]*self.alpha_choice[participant_id.unsqueeze(1), experiment_id]),
            ) 
        }
        
        for module in self.get_modules():
            self.sindy_coefficients[module].requires_grad = False
            self.sindy_coefficients[module].data[participant_id.unsqueeze(1), experiment_id] = torch.nn.Parameter(torch.zeros_like(self.sindy_coefficients[module][participant_id.unsqueeze(1), experiment_id]))
            for candidate_term, value in coefficient_maps[module]:
                self.sindy_coefficients[module].data[participant_id.unsqueeze(1), experiment_id, 0, self.sindy_candidate_terms[module].index(candidate_term)] = value

    def eval(self, *args, **kwargs):
        super().eval()
        self.use_sindy=True
        return self
        
    def train(self, mode=True):
        super().train(mode)
        self.use_sindy=True
        return self
    