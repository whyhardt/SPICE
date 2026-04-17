import torch

from spice import BaseModel, SpiceConfig


spice_config = SpiceConfig(
    library_setup={
        'reward_environment': (
            'reward[t]',
            # 'reward[t-1]',
            # 'reward[t-2]',
            ),
        'reward_patch_harvest': (
            'reward[t]',
            # 'reward[t-1]',
            # 'reward[t-2]',
            ),
        'reward_patch_exit': (
            # 'reward[t-1]',
            # 'reward[t-2]',
            ),
        'depletion_patch_harvest': (
            'dreward[t]', 
            # 'dreward[t-1]',
            # 'dreward[t-2]',
            ),
        'depletion_patch_exit': (
            # 'dreward[t-1]',
            # 'dreward[t-2]',
            ),
        'continuation_patch': (
            'action[t-1]',
            # 'action[t-2]',
            ),
        # 'continuation_patch_harvest': (
        #     'action[t-1]',
        #     # 'action[t-2]',
        #     ),
        # 'continuation_patch_exit': (
        #     # 'action[t-1]',
        #     # 'action[t-2]',
        #     ),
    },
    memory_state={
        'value_reward_environment': 0.5,
        'value_reward_patch': 0.5,
        'value_depletion_patch': 0,
        'value_continuation_patch': 0,
        'reward[t-1]': 1,
        'action[t-1]': 0,
    },
    states_in_logit=(
        'value_reward_environment',
        'value_reward_patch',
        'value_depletion_patch',
        'value_continuation_patch',
    ),
)


class SpiceModel(BaseModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.dropout = 0.1
        
        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )
        
        self.setup_module(key_module='reward_environment', input_size=1, embedding_size=self.embedding_size, include_state=True)
        self.setup_module(key_module='reward_patch_harvest', input_size=1, embedding_size=self.embedding_size, include_state=True)
        self.setup_module(key_module='reward_patch_exit', input_size=0, embedding_size=self.embedding_size, include_state=True)

        self.setup_module(key_module='depletion_patch_harvest', input_size=1, embedding_size=self.embedding_size, include_state=True)
        self.setup_module(key_module='depletion_patch_exit', input_size=0, embedding_size=self.embedding_size, include_state=True)

        self.setup_module(key_module='continuation_patch', input_size=1, embedding_size=self.embedding_size, include_state=True)        
        
    def forward(self, inputs, prev_state=None):
        
        spice_signals = self.init_forward_pass(inputs, prev_state)
        
        # used to update only the action value for harvesting. the action value for exit is kept at 0 as a reference point like in the MVT. 
        mask_harvest_value = torch.zeros_like(spice_signals.actions[0])
        mask_harvest_value[..., 0] = 1
        
        action_harvest = spice_signals.actions[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        action_exit = spice_signals.actions[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        rewards = spice_signals.rewards[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
                
        for trial in spice_signals.trials:
            
            harvested = action_harvest[trial] * mask_harvest_value
            exited = action_exit[trial] * mask_harvest_value

            # 1. Reward processing
            # learning average reward value of environment (only information accumulation; no exit information)
            self.call_module(
                key_module='reward_environment',
                key_state='value_reward_environment',
                action_mask=harvested,
                inputs=(
                    rewards[trial],
                    # self.state['reward[t-1]'],
                    # self.state['reward[t-2]'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # learning current patch reward → patch valuation
            self.call_module(
                key_module='reward_patch_harvest',
                key_state='value_reward_patch',
                action_mask=harvested,
                inputs=(
                    rewards[trial],
                    # self.state['reward[t-1]'],
                    # self.state['reward[t-2]'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='reward_patch_exit',
                key_state='value_reward_patch',
                action_mask=exited,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            
            # 2. Depletion processing
            # depletion mask: only update if previous action was harvest (valid dreward)
            # prev_was_harvest = (self.state['action[t-1]'][..., 0:1] > 0.5).float().expand_as(harvested)
            # mask_depletion_harvest = harvested * prev_was_harvest
            
            # learning current patch depletion → exhaustion signal
            self.call_module(
                key_module='depletion_patch_harvest',
                key_state='value_depletion_patch',
                action_mask=harvested,#mask_depletion_harvest,
                inputs=(
                    rewards[trial]-self.state['reward[t-1]'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='depletion_patch_exit',
                key_state='value_depletion_patch',
                action_mask=exited,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            
            # 3. Continuation effect (e.g. pressure to exit to get to a fresh patch; reward/depletion agnostic)
            self.call_module(
                key_module='continuation_patch',
                key_state='value_continuation_patch',
                action_mask=None,
                inputs=(
                    spice_signals.actions[trial],
                    # self.state['action[t-1]'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # self.call_module(
            #     key_module='continuation_patch_exit',
            #     key_state='value_continuation_patch',
            #     action_mask=exited,
            #     inputs=None,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )
            
            
            # 4. Heuristical memory state updating
            
            # prev_reward: store current reward if harvested, reset to -1 otherwise
            # (only harvesting yields a reward; exiting does not)
            self.state['reward[t-1]'] = torch.where(harvested > 0.5, rewards[trial], 1)
            
            # prev_action: store current trial's action for use in the next trial
            self.state['action[t-1]'] = spice_signals.actions[trial]

            # working memory updates
            # self.state['reward[t-2]'] = torch.where(harvested>0.5, self.state['reward[t-1]'], 0)
            # self.state['reward[t-1]'] = torch.where(harvested>0.5, reward, 0)
            
            # 4. Logit computation
            spice_signals.logits[trial] = (
                + self.state['value_reward_environment']
                + self.state['value_reward_patch']
                + self.state['value_depletion_patch']
                + self.state['value_continuation_patch']
            )
        
        spice_signals = self.post_forward_pass(spice_signals)
        
        return spice_signals.logits, self.get_state()