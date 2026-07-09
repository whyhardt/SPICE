import torch
from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Per-dimension partner influence (4 partner indicators + rank_diff each)
        'module_action': ['partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches', 'rank_diff'],
        'module_grooming': ['partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches', 'rank_diff'],
        'module_gesture': ['partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches', 'rank_diff'],
        'module_scratch': ['partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches', 'rank_diff'],
        # Per-dimension own-action transition (4 own-action indicators + rank_diff each)
        'transition_action': ['own_acts', 'own_grooms', 'own_gestures', 'own_scratches', 'rank_diff'],
        'transition_grooming': ['own_acts', 'own_grooms', 'own_gestures', 'own_scratches', 'rank_diff'],
        'transition_gesture': ['own_acts', 'own_grooms', 'own_gestures', 'own_scratches', 'rank_diff'],
        'transition_scratch': ['own_acts', 'own_grooms', 'own_gestures', 'own_scratches', 'rank_diff'],
        # Self-persistence (rank_diff only)
        'self_repeat': ['rank_diff'],
        'self_switch': ['rank_diff'],
    },
    memory_state={
        'value_partner': 0,
        'value_transition': 0,
        'value_persistence': 0,
    },
    states_in_logit=['value_partner', 'value_transition', 'value_persistence'],
    additional_inputs=('SigAct_ID2', 'ID1', 'ID2', 'Dominance_rank_ID1', 'Dominance_rank_ID2'),
)

# Binary one-hot indicator signals (x^2 = x, mutually exclusive cross-products = 0)
BINARY_SIGNALS = {'partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches',
                  'own_acts', 'own_grooms', 'own_gestures', 'own_scratches'}
EXCLUSIVE_GROUPS = [
    {'partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches'},
    {'own_acts', 'own_grooms', 'own_gestures', 'own_scratches'},
]


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # participant embedding for sender (ID1) only — no experiment embedding
        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

        # per-dimension partner modules (5 control signals each: 4 indicators + rank_diff)
        self.setup_module(key_module='module_action', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='module_grooming', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='module_gesture', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='module_scratch', embedding_size=self.embedding_size, dropout=self.dropout)

        # per-dimension own-action transition modules (5 control signals each)
        self.setup_module(key_module='transition_action', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='transition_grooming', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='transition_gesture', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='transition_scratch', embedding_size=self.embedding_size, dropout=self.dropout)

        # persistence modules (1 control signal: rank_diff)
        self.setup_module(key_module='self_repeat', embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='self_switch', embedding_size=self.embedding_size, dropout=self.dropout)

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out redundant SINDy terms for binary indicator control signals.

        Binary one-hot indicators satisfy x^2 = x and x_i * x_j = 0 for
        mutually exclusive pairs. Continuous signals (rank_diff) are NOT filtered.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            if not control_signals:
                continue
            binary_cs = [cs for cs in control_signals if cs in BINARY_SIGNALS]
            for ict, ct in enumerate(candidate_terms[module]):
                # remove squared terms for binary signals only (x^2 = x)
                for cs in binary_cs:
                    if cs + '^' in ct:
                        self.sindy_coefficients_presence[module][..., ict] = 0
                        self.sindy_coefficients_prior_mask[module][..., ict] = 0
                        break
                # remove cross-products of mutually exclusive indicators (x_i * x_j = 0)
                for group in EXCLUSIVE_GROUPS:
                    group_cs = [cs for cs in control_signals if cs in group]
                    for i, cs_i in enumerate(group_cs):
                        for cs_j in group_cs[i + 1:]:
                            if cs_i in ct and cs_j in ct:
                                self.sindy_coefficients_presence[module][..., ict] = 0
                                self.sindy_coefficients_prior_mask[module][..., ict] = 0

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # participant embedding (sender only)
        embedding = self.participant_embedding(spice_signals.participant_ids)

        # rank difference: (rank_ID1 - rank_ID2), already normalized to [-1, 1] in get_dataset
        rank_id1 = spice_signals.additional_inputs['Dominance_rank_ID1'].squeeze(-1)
        rank_id2 = spice_signals.additional_inputs['Dominance_rank_ID2'].squeeze(-1)
        rank_diff = (rank_id1 - rank_id2).unsqueeze(-1).expand_as(spice_signals.actions)

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
        partner_scratches = actions_id2[..., 3].unsqueeze(-1).expand_as(spice_signals.actions)

        # fixed per-dimension masks (5 actions: action, grooming, gesture, scratch, waiting)
        mask_action = torch.tensor((1, 0, 0, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_grooming = torch.tensor((0, 1, 0, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_gesture = torch.tensor((0, 0, 1, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_scratch = torch.tensor((0, 0, 0, 1, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)

        for timestep in spice_signals.trials:

            # --- own-action indicators for transition modules ---
            own_acts = spice_signals.actions[timestep][..., 0].unsqueeze(-1).expand_as(spice_signals.actions[timestep])
            own_grooms = spice_signals.actions[timestep][..., 1].unsqueeze(-1).expand_as(spice_signals.actions[timestep])
            own_gestures = spice_signals.actions[timestep][..., 2].unsqueeze(-1).expand_as(spice_signals.actions[timestep])
            own_scratches = spice_signals.actions[timestep][..., 3].unsqueeze(-1).expand_as(spice_signals.actions[timestep])

            partner_inputs = (partner_acts[timestep], partner_grooms[timestep], partner_gestures[timestep], partner_scratches[timestep], rank_diff[timestep])
            own_inputs = (own_acts, own_grooms, own_gestures, own_scratches, rank_diff[timestep])

            # --- partner influence: each module updates one action dimension ---
            self.call_module(
                key_module='module_action',
                key_state='value_partner',
                action_mask=mask_action,
                inputs=partner_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='module_grooming',
                key_state='value_partner',
                action_mask=mask_grooming,
                inputs=partner_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='module_gesture',
                key_state='value_partner',
                action_mask=mask_gesture,
                inputs=partner_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='module_scratch',
                key_state='value_partner',
                action_mask=mask_scratch,
                inputs=partner_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            # --- own-action transitions: each module updates one action dimension ---
            self.call_module(
                key_module='transition_action',
                key_state='value_transition',
                action_mask=mask_action,
                inputs=own_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='transition_grooming',
                key_state='value_transition',
                action_mask=mask_grooming,
                inputs=own_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='transition_gesture',
                key_state='value_transition',
                action_mask=mask_gesture,
                inputs=own_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='transition_scratch',
                key_state='value_transition',
                action_mask=mask_scratch,
                inputs=own_inputs,
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            # --- self-persistence: chosen action persists, unchosen decays ---
            self.call_module(
                key_module='self_repeat',
                key_state='value_persistence',
                action_mask=spice_signals.actions[timestep],
                inputs=(rank_diff[timestep],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
            )

            self.call_module(
                key_module='self_switch',
                key_state='value_persistence',
                action_mask=1 - spice_signals.actions[timestep],
                inputs=(rank_diff[timestep],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=embedding,
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
