import torch


from spice import SpiceConfig, BaseModel, SpiceDataset


CONFIG = SpiceConfig(
    library_setup={
        'perception_certainty': ['contr_diff[t]'],
        'value_wm_reward_chosen': ['reward[t]', 'certainty[t]'],
        'value_wm_reward_not_chosen': ['certainty[t]'],
        'value_reward_chosen': ['reward[t]', 'certainty[t]'],
        'value_reward_not_chosen': ['certainty[t]'],
        'value_choice': ['choice[t]', 'certainty[t]', 'certainty_next[t+1]'],
        # 'value_choice_chosen': ['certainty[t]', 'certainty_next[t+1]'],
        # 'value_choice_not_chosen': ['certainty[t]', 'certainty_next[t+1]'],
    },
    memory_state={
        'value_wm_reward': 0,
        'value_reward': 0,
        'value_choice': 0,
    },
    states_in_logit=[
        'value_wm_reward',
        'value_reward',
        'value_choice',
    ],
    additional_inputs=('contrast_difference', 'contrast_difference_next'),
)


# Binary indicator control signals (x^2 = x): reward is 0/1.
BINARY_SIGNALS = {'reward[t]', 'choice[t]'}


def prepare_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """Append the next trial's contrast difference as an additional input.

    'contrast_difference_next' is not a column of the CSV: it is the within-session
    lead of 'contrast_difference'. Building it here costs the last trial of each
    session, which is dropped from both xs and ys.
    """
    n_actions = dataset.ys.shape[-1]
    contr_diff = dataset.xs[..., n_actions * 2].unsqueeze(-1)
    contr_diff_next = contr_diff[:, 1:]
    xs = torch.cat((
        dataset.xs[:, :-1, :, :n_actions * 2],
        contr_diff[:, :-1, :],
        contr_diff_next,
        dataset.xs[:, :-1, :, 2 * n_actions + 1:],
    ), dim=-1)
    return SpiceDataset(xs, dataset.ys[:, :-1])


class SpiceModel(BaseModel):
    
    def __init__(self, deterministic_perception=False, **kwargs):
        super().__init__(**kwargs)
        
        self.deterministic_perception = deterministic_perception
        
        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )
        
        # perception: signed contr_diff → sigmoid
        self.setup_module(key_module='perception_certainty', include_state=False)
        self.setup_module(key_module='value_wm_reward_chosen', include_state=False)

        self.preprocess_coefficients()
        
    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for binary indicators.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            binary_signals = [s for s in self.spice_config.library_setup[module] if s in BINARY_SIGNALS]
            for index_term, term in enumerate(candidate_terms[module]):
                factors = term.split('*')
                if any(signal + '^2' in factors for signal in binary_signals):
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

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

            # --- PERCEPTION: certainty about the current trial's item assignment ---
            certainty_current = torch.nn.functional.sigmoid(self.call_module(
                key_module='perception_certainty',
                inputs=contr_diff_current[trial].abs(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            ))

            # Item-space hard masks use the deterministic contrast mapping; certainty is an
            # input so updates can be attenuated when that assignment is unreliable.

            # --- WORKING MEMORY UPDATES ---
            self.call_module(
                key_module='value_wm_reward_chosen',
                key_state='value_wm_reward',
                action_mask=action_contrast[trial],
                inputs=(
                    reward_scalar[trial],
                    certainty_current,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            self.call_module(
                key_module='value_wm_reward_not_chosen',
                key_state='value_wm_reward',
                action_mask=1 - action_contrast[trial],
                inputs=(certainty_current,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # --- REWARD VALUE UPDATES ---
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=action_contrast[trial],
                inputs=(
                    reward_scalar[trial],
                    certainty_current,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - action_contrast[trial],
                inputs=(certainty_current,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # --- PERCEPTION: certainty about the next trial's item assignment ---
            certainty_next = torch.nn.functional.sigmoid(self.call_module(
                key_module='perception_certainty',
                inputs=contr_diff_next[trial].abs(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            ))

            # --- CHOICE VALUE UPDATES ---
            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                inputs=(
                    action_contrast[trial],
                    certainty_current,
                    certainty_next,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )
            
            # self.call_module(
            #     key_module='value_choice_chosen',
            #     key_state='value_choice',
            #     action_mask=action_contrast[trial],
            #     inputs=(
            #         certainty_current,
            #         certainty_next,
            #     ),
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embeddings,
            # )

            # self.call_module(
            #     key_module='value_choice_not_chosen',
            #     key_state='value_choice',
            #     action_mask=1 - action_contrast[trial],
            #     inputs=(
            #         certainty_current,
            #         certainty_next,
            #     ),
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embeddings,
            # )

            # --- LOGITS ---
            logits_item_space = (
                self.state['value_reward']
                + self.state['value_wm_reward']
                + self.state['value_choice']
            )

            # p(assignment correct) from contrast-difference-based certainty; range=(0.5, 1.0)
            certainty_next = certainty_next / 2 + 0.5
            mixed_logits_item_space = logits_item_space * certainty_next + logits_item_space.flip(-1) * (1 - certainty_next)

            # map mixed logits from item space (low, high) into action space (left, right)
            spice_signals.logits[trial] = torch.where(cd_next[trial] < 0, mixed_logits_item_space, mixed_logits_item_space.flip(-1))

        spice_signals = self.post_forward_pass(spice_signals)
        
        return spice_signals.logits, self.get_state()