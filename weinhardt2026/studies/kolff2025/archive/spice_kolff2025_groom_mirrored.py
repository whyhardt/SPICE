import torch

from spice import SpiceConfig, BaseModel, SpiceDataset

from .spice_kolff2025_groom_id2 import N_ACTIONS, masked_cross_entropy


# -------------------------------------------------------------------------------------------
# GROOMING MODEL -- BOTH PERSPECTIVES (MIRRORED DATA)
# -------------------------------------------------------------------------------------------
# Every interaction is read twice (benchmarking_kolff2025_groom_id2.build_mirrored_dataframe):
# from the approached ape's (ID2's) and from the approacher's (ID1's) point of view. The
# participant is the focal ape; predicts whether its next act is grooming (item 1) or not
# (item 0, reference logit 0), scored only on events where it acts next.
#
# Structure as in spice_kolff2025_groom_id2_shared, plus the flag `approacher` (1 = the
# participant is ID1) as an input of every module: with degree 2 the library holds
# approacher x signal terms, i.e. role-specific effects where the data need them.
#   value_negotiation       -- negotiation categories of partner and self (own coefficients)
#   value_grooming_received -- the partner's preparation and grooming acts
#   value_grooming_given    -- the participant's own preparation and grooming acts
# Every module uses its state in its update; the grooming logit is the sum of the three states.
# -------------------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        'negotiation': [
            'grooming_solicitation',
            'self_reposition',
            'directed_scratch',
            'own_grooming_solicitation',
            'own_self_reposition',
            'own_directed_scratch',
            'rank_difference',
            'approacher',
            ],
        'grooming_received': [
            'reposition_body',
            'grooming_process',
            'grooming',
            'rank_difference',
            'approacher',
            ],
        'grooming_given': [
            'own_reposition_body',
            'own_grooming_process',
            'own_grooming',
            'rank_difference',
            'approacher',
            ],
        },
    memory_state={
        'value_negotiation': 0,
        'value_grooming_received': 0,
        'value_grooming_given': 0,
    },
    states_in_logit=[
        'value_negotiation',
        'value_grooming_received',
        'value_grooming_given',
        ],
    additional_inputs=(
        'partner_grooming_solicitation',
        'partner_self_reposition',
        'partner_directed_scratch',
        'partner_reposition_body',
        'partner_grooming_process',
        'partner_groom',
        'self_grooming_solicitation',
        'self_self_reposition',
        'self_directed_scratch',
        'self_reposition_body',
        'self_grooming_process',
        'self_groom',
        'rank_diff_centered',
        'is_approacher',
        'self_acts_next',  # target mask only, never a model input (future information)
        ),
)

# category indicators and the role flag are binary (x^2 = x); an ape acts with at most one
# category per event (x_i * x_j = 0 within an ape), and partner and self act together in
# only 5 of 5905 events, so those products are excluded in the negotiation module as well
BINARY_SIGNALS = (
    'grooming_solicitation', 'self_reposition', 'directed_scratch', 'reposition_body', 'grooming_process', 'grooming',
    'own_grooming_solicitation', 'own_self_reposition', 'own_directed_scratch', 'own_reposition_body',
    'own_grooming_process', 'own_grooming', 'approacher',
)
EXCLUSIVE_GROUPS = [
    {'grooming_solicitation', 'self_reposition', 'directed_scratch',
     'own_grooming_solicitation', 'own_self_reposition', 'own_directed_scratch'},
    {'reposition_body', 'grooming_process', 'grooming'},
    {'own_reposition_body', 'own_grooming_process', 'own_grooming'},
]

COL_TARGET_MASK = N_ACTIONS + CONFIG.additional_inputs.index('self_acts_next')
COL_APPROACHER = N_ACTIONS + CONFIG.additional_inputs.index('is_approacher')


def target_mask(ys: torch.Tensor) -> torch.Tensor:
    """True where the participant acts at the next event, i.e. where a target exists."""
    return (ys[..., N_ACTIONS] == 1) & ~torch.isnan(ys[..., 0])


def prepare_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """Pack the target mask into ys: ys = [not groom, groom, participant acts next]."""
    mask = dataset.xs[..., COL_TARGET_MASK:COL_TARGET_MASK + 1].clone()
    return SpiceDataset(dataset.xs, torch.cat((dataset.ys, mask), dim=-1), n_reward_features=0)


# Hooks read by weinhardt2026/run.py
NORMALIZE_REWARDS = False       # no reward columns
LOSS_FN = masked_cross_entropy
LOSS_FN_KWARGS = {'label_smoothing': 0.01}
ESTIMATOR_KWARGS = {'n_actions': N_ACTIONS, 'n_items': N_ACTIONS}   # ys carries the target mask as a 3rd column


class SpiceModel(BaseModel):
    """Shared negotiation state plus one grooming state per ape, with the approacher role as an input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for binary indicators.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        Two indicators of the same mutually exclusive group satisfy x_i * x_j = 0.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            binary_signals = [s for s in control_signals if s in BINARY_SIGNALS]
            for index_term, term in enumerate(candidate_terms[module]):
                factors = term.split('*')
                redundant = any(signal + '^2' in factors for signal in binary_signals)
                for group in EXCLUSIVE_GROUPS:
                    redundant |= sum(signal in factors for signal in group if signal in control_signals) > 1
                if redundant:
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        index_groom = 1
        mask_groom = torch.zeros_like(spice_signals.logits[0])
        mask_groom[..., index_groom] = 1

        for trial in spice_signals.trials:

            signals = spice_signals.additional_inputs
            rank_difference = signals['rank_diff_centered'][trial]
            approacher = signals['is_approacher'][trial]

            self.call_module(
                key_module='negotiation',
                key_state='value_negotiation',
                action_mask=mask_groom,
                inputs=(
                    signals['partner_grooming_solicitation'][trial],
                    signals['partner_self_reposition'][trial],
                    signals['partner_directed_scratch'][trial],
                    signals['self_grooming_solicitation'][trial],
                    signals['self_self_reposition'][trial],
                    signals['self_directed_scratch'][trial],
                    rank_difference,
                    approacher,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='grooming_received',
                key_state='value_grooming_received',
                action_mask=mask_groom,
                inputs=(
                    signals['partner_reposition_body'][trial],
                    signals['partner_grooming_process'][trial],
                    signals['partner_groom'][trial],
                    rank_difference,
                    approacher,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='grooming_given',
                key_state='value_grooming_given',
                action_mask=mask_groom,
                inputs=(
                    signals['self_reposition_body'][trial],
                    signals['self_grooming_process'][trial],
                    signals['self_groom'][trial],
                    rank_difference,
                    approacher,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            spice_signals.logits[trial] = \
                + self.state['value_negotiation'] \
                + self.state['value_grooming_received'] \
                + self.state['value_grooming_given']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
