import torch


from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        'perception_certainty': ['contr_diff[t]'],
        'reward_learning_chosen': ['reward[t]', 'certainty[t]'],
        'reward_learning_unchosen': ['certainty[t]'],
        'choice_persistance': ['choice[t]', 'certainty[t]', 'certainty_next[t+1]'],
    },
    
    memory_state={
        'value_reward_contrast': None,
        'value_choice_contrast': None,
    },
    additional_inputs=('contrast_difference', 'contrast_difference_next'),
)


class SpiceModel(BaseModel):
    
    def __init__(self, deterministic_perception=False, **kwargs):
        super().__init__(**kwargs)
        
        self.dropout = 0.1
        self.deterministic_perception = deterministic_perception
        
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout)
        
        # perception: signed contr_diff → sigmoid
        self.setup_module(key_module='perception_certainty', input_size=1, include_state=False, dropout=self.dropout)
        # chosen item: learns from reward, modulated by certainty
        self.setup_module(key_module='reward_learning_chosen', input_size=2, dropout=self.dropout)
        # unchosen item: forgetting/persistence dynamics, modulated by certainty
        self.setup_module(key_module='reward_learning_unchosen', input_size=1, dropout=self.dropout)
        # choice persistance module
        self.setup_module(key_module='choice_persistance', input_size=3, dropout=self.dropout)
        
    def forward(self, inputs, state = None):
        spice_signals = self.init_forward_pass(inputs, state)
        
        # feature extraction
        cd_current = spice_signals.additional_inputs['contrast_difference'].squeeze(-1)  # (T, W, E, B)
        cd_next = spice_signals.additional_inputs['contrast_difference_next']            # (T, W, E, B, 1)
        # repeated to n_actions for module inputs (T, W, E, B, n_actions)
        contr_diff_current = cd_current.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_actions)
        contr_diff_next = cd_next.repeat(1, 1, 1, 1, self.n_actions)
        
        # Map actions: position space (left=0, right=1) → item space (low=0, high=1)
        # cd <= 0: left=low, right=high;  cd > 0: left=high, right=low
        chose = spice_signals.actions.argmax(dim=-1)  # (T, W, E, B)
        action_contrast = torch.zeros_like(spice_signals.actions)
        # chose_low: chose left when left=low (cd<=0), OR chose right when right=low (cd>0)
        action_contrast[..., 0] = (((cd_current <= 0) & (chose == 0)) | ((cd_current > 0) & (chose == 1))).float()
        # chose_high: chose right when right=high (cd<=0), OR chose left when left=high (cd>0)
        action_contrast[..., 1] = (((cd_current <= 0) & (chose == 1)) | ((cd_current > 0) & (chose == 0))).float()
        
        # Scalar reward per trial (sum over one-hot reward vector)
        reward_scalar = spice_signals.feedback.sum(dim=-1, keepdim=True).expand_as(spice_signals.actions)
        
        participant_embeddings = self.participant_embedding(spice_signals.participant_ids)
        
        for trial in spice_signals.trials:
            
            # --- Perception: p(left=low) for current trial ---
            certainty_current = self.call_module(
                key_module='perception_certainty',
                inputs=contr_diff_current[trial],#.abs(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
                activation_rnn=torch.nn.functional.sigmoid,
                # activation_rnn=torch.nn.functional.tanh,
            )
            
            certainty_next = self.call_module(
                key_module='perception_certainty',
                inputs=contr_diff_next[trial],#.abs(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
                activation_rnn=torch.nn.functional.sigmoid,
                # activation_rnn=torch.nn.functional.tanh,
            )

            # --- Reward learning with explicit chosen/unchosen split ---
            # Hard masks use the deterministic item space action (ground truth mapping).
            # Certainty is passed as input so the GRU can learn to attenuate updates
            # when the deterministic assignment is unreliable.
            
            # Chosen item: update value based on actual reward
            self.call_module(
                key_module='reward_learning_chosen',
                key_state='value_reward_contrast',
                action_mask=action_contrast[trial],
                inputs=(
                    reward_scalar[trial], 
                    certainty_current,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )
            
            # Unchosen item: can learn forgetting, persistence, or drift
            self.call_module(
                key_module='reward_learning_unchosen',
                key_state='value_reward_contrast',
                action_mask=1 - action_contrast[trial],
                inputs=(certainty_current,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )
            
            # Choice persistance
            self.call_module(
                key_module='choice_persistance',
                key_state='value_choice_contrast',
                inputs=(
                    action_contrast[trial],
                    certainty_current,
                    certainty_next,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )
            
            # --- Decision: soft-map item values to action space ---
                        
            logits_contrast = self.state['value_reward_contrast'] + self.state['value_choice_contrast']
            
            # deriving p(left=low) from contrast-difference-based certainty to assign learned reward values to given options; range=(0.5, 1.0)
            # TODO: add sign information about contrast difference
            # certainty_next = (certainty_next + 1) / 2
            certainty_next = certainty_next / 2 + 0.5
            mixed_logits_item_space = logits_contrast * certainty_next + logits_contrast.flip(-1) * (1 - certainty_next)
            
            # map mixed logits from item space (low, high) into action space (left, right)
            spice_signals.logits[trial] = torch.where(cd_next[trial] < 0, mixed_logits_item_space, mixed_logits_item_space.flip(-1))
            
        spice_signals = self.post_forward_pass(spice_signals)
        
        return spice_signals.logits, self.get_state()