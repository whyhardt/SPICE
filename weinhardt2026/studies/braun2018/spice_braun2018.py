import torch

from spice import SpiceConfig, BaseModel


"""
Thoughts on SPICE architecture:

1. Reward and Effort sensitivity -> Separate values for reward and effort
2. Include effort levels for both tasks
3. Item-Space -> Task-Space -> Enables tracking task-individual values (e.g. some people might find easier to say color and others prefer form) 
4. Action-Space -> Repeat/Switch or specific Task-Selection?
5. Include separate module for repeat/switch bias -> May be unncessary because could be encoded already in effort modules
6. Include previous response time as input to repeat/switch bias
7. Account for length of experiment as cognitive control becomes more costly over time i.e. blocks ("We predict that the indifference point will increase as the experiment progresses")
8. Check whether state-less implementation makes sense -> Everything is given in cues (rewards for each task; effort levels could be directly predicted using last n actions) vs state-based dynamical model (evolution of effort/reward values over trials)

On repeat/switch bias:

"control becomes increasingly costly as it is exerted for a longer period of time" - Braun (2018)
-> Switching to keep optimal reward strength cannot be done over an extended period
-> Repeat/Switch ratio is probabily very unbalanced while Task-Ratio should be more or less counterbalanced -> **Benefit of acting on Task-Space instead of Repeat/Switch-Space**
"""


BINARY_SIGNALS = {'repeat'}


CONFIG = SpiceConfig(
    library_setup={
        'reward_repeat': (
            'dreward_tasks',
            # 'dreward_trials_repeat',
            # 'dreward_trials_switch',
            ),
        'reward_switch': (
            'dreward_tasks',
            # 'dreward_trials_repeat',
            # 'dreward_trials_switch',
            ),
        'task_repeat': (
            'repeat',
            ),
        'task_switch': (
            'repeat',
            ),
        'fatigue_repeat': (
            'block',
            ),
        'fatigue_switch': (
            'block',
            ),
    },
    memory_state={
        'value_reward': None,
        'value_control': None,
        'value_fatigue': None,
    },
    additional_inputs=('difference', 'current', 'other'),
)


class SpiceModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.dropout = 0.1
        self.n_blocks = 12
        
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout)
        
        self.setup_module(key_module='reward_repeat', input_size=1, dropout=self.dropout, include_state=False)
        self.setup_module(key_module='reward_switch', input_size=1, dropout=self.dropout, include_state=False)
        self.setup_module(key_module='task_repeat', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='task_switch', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='fatigue_repeat', input_size=1, dropout=self.dropout, include_state=False)
        self.setup_module(key_module='fatigue_switch', input_size=1, dropout=self.dropout, include_state=False)

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out redundant SINDy terms for binary indicator control signals.

        Binary indicators satisfy x^2 = x, so squared terms are redundant.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            if not control_signals:
                continue
            binary_cs = [cs for cs in control_signals if cs in BINARY_SIGNALS]
            for ict, ct in enumerate(candidate_terms[module]):
                for cs in binary_cs:
                    if cs + '^' in ct:
                        self.sindy_coefficients_presence[module][..., ict] = 0
                        self.sindy_coefficients_prior_mask[module][..., ict] = 0
                        break

    def forward(self, inputs, prev_state=None):
        
        spice_signals = self.init_forward_pass(inputs, prev_state)
        
        blocks = spice_signals.blocks.unsqueeze(0).unsqueeze(-1).expand_as(spice_signals.actions[0]) / self.n_blocks
        
        repeat = torch.zeros_like(spice_signals.actions)
        repeat += spice_signals.actions[..., :1]
        switch = torch.zeros_like(spice_signals.actions)
        switch += spice_signals.actions[..., 1:2]
        
        repeat_mask = torch.zeros_like(spice_signals.actions[0])
        repeat_mask[..., 0] = 1
        switch_mask = torch.zeros_like(spice_signals.actions[0])
        switch_mask[..., 1] = 1
        
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        
        for trial in spice_signals.trials:
            
            # modules for reward perception
            self.call_module(
                key_module='reward_repeat',
                key_state='value_reward',
                action_mask=repeat_mask,
                inputs=-spice_signals.additional_inputs['difference'][trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='reward_switch',
                key_state='value_reward',
                action_mask=switch_mask,
                inputs=spice_signals.additional_inputs['difference'][trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # modules to update switching costs
            self.call_module(
                key_module='task_repeat',
                key_state='value_control',
                action_mask=repeat_mask,
                inputs=repeat[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='task_switch',
                key_state='value_control',
                action_mask=switch_mask,
                inputs=repeat[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # module to update fatigue value based on current block number
            self.call_module(
                key_module='fatigue_repeat',
                key_state='value_fatigue',
                action_mask=repeat_mask,
                inputs=blocks,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='fatigue_switch',
                key_state='value_fatigue',
                action_mask=switch_mask,
                inputs=blocks,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # compute logits
            spice_signals.logits[trial] = (
                self.state['value_reward']
                + self.state['value_control']
                + self.state['value_fatigue']
                )
        
        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.state
