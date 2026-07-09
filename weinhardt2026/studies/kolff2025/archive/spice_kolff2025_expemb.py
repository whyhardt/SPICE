import torch
from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Per-dimension partner influence (3 partner indicators each)
        'module_action': ['partner_acts', 'partner_grooms', 'partner_gestures'],
        'module_grooming': ['partner_acts', 'partner_grooms', 'partner_gestures'],
        'module_gesture': ['partner_acts', 'partner_grooms', 'partner_gestures'],
        # Per-dimension own-action transition (3 own-action indicators each)
        'transition_action': ['own_acts', 'own_grooms', 'own_gestures'],
        'transition_grooming': ['own_acts', 'own_grooms', 'own_gestures'],
        'transition_gesture': ['own_acts', 'own_grooms', 'own_gestures'],
        # Self-persistence (no control signals, just state + embedding)
        'self_repeat': [],
        'self_switch': [],
    },
    memory_state={
        'value_partner': 0,
        'value_transition': 0,
        'value_persistence': 0,
    },
    states_in_logit=['value_partner', 'value_transition', 'value_persistence'],
    additional_inputs=('SigAct_ID2', 'ID1', 'ID2'),
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # shared embedding table for all chimps (sender and receiver looked up from same table)
        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
            target_embedding_size_fusion=8,
        )

        # per-dimension partner modules (3 control signals each)
        self.setup_module(key_module='module_action', embedding_size=self.embedding_size * 2, dropout=self.dropout)
        self.setup_module(key_module='module_grooming', embedding_size=self.embedding_size * 2, dropout=self.dropout)
        self.setup_module(key_module='module_gesture', embedding_size=self.embedding_size * 2, dropout=self.dropout)

        # per-dimension own-action transition modules (3 control signals each)
        self.setup_module(key_module='transition_action', embedding_size=self.embedding_size * 2, dropout=self.dropout)
        self.setup_module(key_module='transition_grooming', embedding_size=self.embedding_size * 2, dropout=self.dropout)
        self.setup_module(key_module='transition_gesture', embedding_size=self.embedding_size * 2, dropout=self.dropout)

        # persistence modules (0 control signals)
        self.setup_module(key_module='self_repeat', embedding_size=self.embedding_size * 2, dropout=self.dropout)
        self.setup_module(key_module='self_switch', embedding_size=self.embedding_size * 2, dropout=self.dropout)

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out redundant SINDy terms for binary indicator control signals.

        All control signals are one-hot indicators, so:
        - x^2 = x for binary signals → squared terms redundant with linear terms
        - x_i * x_j = 0 for mutually exclusive indicators → cross-products always zero
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            if not control_signals:
                continue
            for ict, ct in enumerate(candidate_terms[module]):
                # remove squared terms (x^2 = x for binary)
                for cs in control_signals:
                    if cs + '^' in ct:
                        self.sindy_coefficients_presence[module][..., ict] = 0
                        self.sindy_coefficients_prior_mask[module][..., ict] = 0
                        break
                # remove cross-products of mutually exclusive indicators (x_i * x_j = 0)
                for i, cs_i in enumerate(control_signals):
                    for cs_j in control_signals[i + 1:]:
                        if cs_i in ct and cs_j in ct:
                            self.sindy_coefficients_presence[module][..., ict] = 0
                            self.sindy_coefficients_prior_mask[module][..., ict] = 0

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # embeddings: sender (participant) and receiver (experiment) from shared table
        embedding_id1 = self.participant_embedding(spice_signals.participant_ids)
        embedding_id2 = self.participant_embedding(spice_signals.experiment_ids)

        # partner (ID2) action indicators — one-hot encode from scalar
        actions_id2 = torch.eye(self.n_actions, device=self.device)[
            spice_signals.additional_inputs['SigAct_ID2'].squeeze(-1).int()
        ]
        # NaN-padded trials become SigAct_ID2=0 after nan_to_num, falsely
        # encoding as "partner did action". Zero out using the action sum:
        # valid trials have one-hot actions (sum=1), padded have all-zero (sum=0).
        padding_mask = spice_signals.actions.sum(dim=-1) == 0
        actions_id2[padding_mask] = 0

        partner_acts = actions_id2[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        partner_grooms = actions_id2[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        partner_gestures = actions_id2[..., 2].unsqueeze(-1).expand_as(spice_signals.actions)

        # fixed per-dimension masks
        mask_action = torch.tensor((1, 0, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_grooming = torch.tensor((0, 1, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_gesture = torch.tensor((0, 0, 1, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)

        for timestep in spice_signals.trials:

            # --- own-action indicators for transition modules ---
            own_acts = spice_signals.actions[timestep][..., 0].unsqueeze(-1).expand_as(spice_signals.actions[timestep])
            own_grooms = spice_signals.actions[timestep][..., 1].unsqueeze(-1).expand_as(spice_signals.actions[timestep])
            own_gestures = spice_signals.actions[timestep][..., 2].unsqueeze(-1).expand_as(spice_signals.actions[timestep])

            # --- partner influence: each module updates one action dimension ---
            self.call_module(
                key_module='module_action',
                key_state='value_partner',
                action_mask=mask_action,
                inputs=(partner_acts[timestep], partner_grooms[timestep], partner_gestures[timestep]),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            self.call_module(
                key_module='module_grooming',
                key_state='value_partner',
                action_mask=mask_grooming,
                inputs=(partner_acts[timestep], partner_grooms[timestep], partner_gestures[timestep]),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            self.call_module(
                key_module='module_gesture',
                key_state='value_partner',
                action_mask=mask_gesture,
                inputs=(partner_acts[timestep], partner_grooms[timestep], partner_gestures[timestep]),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            # --- own-action transitions: each module updates one action dimension ---
            self.call_module(
                key_module='transition_action',
                key_state='value_transition',
                action_mask=mask_action,
                inputs=(own_acts, own_grooms, own_gestures),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            self.call_module(
                key_module='transition_grooming',
                key_state='value_transition',
                action_mask=mask_grooming,
                inputs=(own_acts, own_grooms, own_gestures),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            self.call_module(
                key_module='transition_gesture',
                key_state='value_transition',
                action_mask=mask_gesture,
                inputs=(own_acts, own_grooms, own_gestures),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            # --- self-persistence: chosen action persists, unchosen decays ---
            self.call_module(
                key_module='self_repeat',
                key_state='value_persistence',
                action_mask=spice_signals.actions[timestep],
                inputs=(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            self.call_module(
                key_module='self_switch',
                key_state='value_persistence',
                action_mask=1 - spice_signals.actions[timestep],
                inputs=(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=embedding_id2,
            )

            # --- logits: additive composition ---
            spice_signals.logits[timestep] = self.state['value_partner'] + self.state['value_transition'] + self.state['value_persistence']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()


def filter_non_waiting(ys: torch.Tensor) -> torch.Tensor:
    """Return (B, T) bool mask that is True for non-waiting trials."""
    return torch.argmax(ys[:, :, 0, :], dim=-1) != (ys.shape[-1] - 1)


def cross_entropy_loss_mask_waiting(prediction, target, label_smoothing: float = 0.):

    n_actions = target.shape[-1]
    waiting_action = n_actions - 1

    prediction = prediction.reshape(-1, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)

    non_waiting_mask = target != waiting_action
    if not non_waiting_mask.any():
        return prediction.sum() * 0.

    return torch.nn.functional.cross_entropy(
        prediction[non_waiting_mask],
        target[non_waiting_mask],
        label_smoothing=label_smoothing,
    )
