import torch

from spice import BaseModel
from spice import SpiceConfig

CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],   # --> n_terms = 6
        'value_reward_not_chosen': [],       # --> n_terms = 3
        'value_choice_chosen': [],           # --> n_terms = 3
        'value_choice_not_chosen': [],       # --> n_terms = 3
        # 'dvalue_reward_chosen': ['dvalue'],  # --> n_terms = 6
        # 'dvalue_reward_not_chosen': [],      # --> n_terms = 6
        # 'dvalue_choice': ['dvalue'],         # --> n_terms = 6
    },                                       # --> n_terms_total = 21
    memory_state={
        'value_reward': 0.,
        'value_choice': 0.,
        # 'dvalue_reward': 0.,
        # 'dvalue_choice': 0.,

        # Buffer (excluded from logits)
        # 'value_reward[t-1]': 0.,
        # 'value_choice[t-1]': 0.,
    },
    states_in_logit=[
        'value_reward',
        'value_choice',
        # 'dvalue_reward',
        # 'dvalue_choice',
    ],
)


class SpiceModel(BaseModel):

    def __init__(self, reward_binary: bool = True, **kwargs):
        super().__init__(**kwargs)

        dropout = 0.1

        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)

        self.preprocess_coefficients(reward_binary=reward_binary)
        
    def forward(self, inputs, prev_state=None):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
        """

        # First, we have to initialize all the inputs and outputs (i.e. logits)
        spice_signals = self.init_forward_pass(inputs, prev_state)

        # We compute now the participant embeddings before the for-loop because they are anyways time-invariant
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for timestep in spice_signals.trials:

            # updates for value_reward
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.feedback[timestep],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
            )

            # updates for value_choice            
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
            )

            # # tracking of the reward-value change of the chosen option
            # dvalue_reward = (self.state['value_reward'] - self.state['value_reward[t-1]']).detach()

            # self.call_module(
            #     key_module='dvalue_reward_chosen',
            #     key_state='dvalue_reward',
            #     action_mask=spice_signals.actions[timestep],
            #     inputs=dvalue_reward,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            #     experiment_index=spice_signals.experiment_ids,
            # )
            
            # self.call_module(
            #     key_module='dvalue_reward_not_chosen',
            #     key_state='dvalue_reward',
            #     action_mask=1-spice_signals.actions[timestep],
            #     inputs=None,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            #     experiment_index=spice_signals.experiment_ids,
            # )
            
            # # buffer update
            # self.state['value_reward[t-1]'] = self.state['value_reward']
            
            # # tracking of the reward-value change of the chosen option
            # dvalue_choice = (self.state['value_choice'] - self.state['value_choice[t-1]']).detach()

            # self.call_module(
            #     key_module='dvalue_choice',
            #     key_state='dvalue_choice',
            #     action_mask=None,
            #     inputs=dvalue_choice,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            #     experiment_index=spice_signals.experiment_ids,
            # )

            # # buffer update
            # self.state['value_reward[t-1]'] = self.state['value_reward']
            # self.state['value_choice[t-1]'] = self.state['value_choice']

            # Now keep track of the logit in the output array
            spice_signals.logits[timestep] = (
                self.state['value_reward'] \
                + self.state['value_choice'] \
                # + self.state['dvalue_reward'] \
                # + self.state['dvalue_choice'] \
            )

        # post-process the forward pass
        spice_signals = self.post_forward_pass(spice_signals)

        return spice_signals.logits, self.get_state()
    
    def preprocess_coefficients(self, reward_binary: bool = True):
        # remove unnecessary candidate terms, e.g. polynomials of binary signals
        # if reward_binary: reward[t] = reward[t]^2 -> presence[reward[t]^2] = 0
        # accounts for ALL control signals in workingmemory model if reward is binary; else only choice signals

        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            if ('reward' in module and reward_binary) or 'choice' in module:
                control_signals = self.spice_config.library_setup[module]
                for cs in control_signals:
                    for ict, ct in enumerate(candidate_terms[module]):
                        if cs+'^' in ct:
                            self.sindy_coefficients_presence[module][..., ict] = 0
                            self.sindy_coefficients_prior_mask[module][..., ict] = 0
