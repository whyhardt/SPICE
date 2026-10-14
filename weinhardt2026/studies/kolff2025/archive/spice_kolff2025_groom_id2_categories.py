import torch

from spice import SpiceConfig, BaseModel, SpiceDataset

from .spice_kolff2025_groom_id2 import N_ACTIONS, target_mask, masked_cross_entropy


# -------------------------------------------------------------------------------------------
# ID2 GROOMING MODEL -- SINGLE CATEGORIES
# -------------------------------------------------------------------------------------------
# ID2 (approached ape) is the participant. Predicts whether ID2's next act is grooming
# (item 1) or not (item 0, reference logit 0); scored only on events where ID2 acts next.
#
# Same six modules as spice_kolff2025_groom_id2 (three for ID1's acts, three for ID2's own
# acts), but each module receives the single categories of its group instead of the group
# indicator. Every module has its own state, which it uses in its update; the grooming logit
# is the sum of the six states.
# -------------------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        # ID1's acts, as received by ID2
        'negotiation_observed': [
            'grooming_solicitation',
            'self_reposition',
            'directed_scratch',
            'rank_difference',
            ],
        'grooming_preparation_received': [
            'reposition_body',
            'grooming_process',
            'rank_difference',
            ],
        'grooming_received': [
            'grooming',
            'rank_difference',
            ],
        # ID2's own acts
        'negotiation_given': [
            'own_grooming_solicitation',
            'own_self_reposition',
            'own_directed_scratch',
            'rank_difference',
            ],
        'grooming_preparation_given': [
            'own_reposition_body',
            'own_grooming_process',
            'rank_difference',
            ],
        'grooming_given': [
            'own_grooming',
            'rank_difference',
            ],
        },
    memory_state={
        'value_negotiation_observed': 0,
        'value_grooming_preparation_received': 0,
        'value_grooming_received': 0,
        'value_negotiation_given': 0,
        'value_grooming_preparation_given': 0,
        'value_grooming_given': 0,
    },
    states_in_logit=[
        'value_negotiation_observed',
        'value_grooming_preparation_received',
        'value_grooming_received',
        'value_negotiation_given',
        'value_grooming_preparation_given',
        'value_grooming_given',
        ],
    additional_inputs=(
        'id1_grooming_solicitation',
        'id1_self_reposition',
        'id1_directed_scratch',
        'id1_reposition_body',
        'id1_grooming_process',
        'id1_groom',
        'id2_grooming_solicitation',
        'id2_self_reposition',
        'id2_directed_scratch',
        'id2_reposition_body',
        'id2_grooming_process',
        'id2_groom',
        'rank_diff_centered',
        'id2_acts_next',  # target mask only, never a model input (future information)
        ),
)

# category indicators are binary (x^2 = x), and an ape acts with at most one category per
# event, so two categories within the same module never co-occur (x_i * x_j = 0)
BINARY_SIGNALS = (
    'grooming_solicitation', 'self_reposition', 'directed_scratch', 'reposition_body', 'grooming_process', 'grooming',
    'own_grooming_solicitation', 'own_self_reposition', 'own_directed_scratch', 'own_reposition_body',
    'own_grooming_process', 'own_grooming',
)
EXCLUSIVE_GROUPS = [
    {'grooming_solicitation', 'self_reposition', 'directed_scratch'},
    {'reposition_body', 'grooming_process'},
    {'own_grooming_solicitation', 'own_self_reposition', 'own_directed_scratch'},
    {'own_reposition_body', 'own_grooming_process'},
]

COL_TARGET_MASK = N_ACTIONS + CONFIG.additional_inputs.index('id2_acts_next')


def prepare_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """Pack the target mask into ys: ys = [not groom, groom, ID2 acts next]."""
    mask = dataset.xs[..., COL_TARGET_MASK:COL_TARGET_MASK + 1].clone()
    return SpiceDataset(dataset.xs, torch.cat((dataset.ys, mask), dim=-1), n_reward_features=0)


# Hooks read by weinhardt2026/run.py
NORMALIZE_REWARDS = False       # no reward columns
LOSS_FN = masked_cross_entropy
LOSS_FN_KWARGS = {'label_smoothing': 0.01}
ESTIMATOR_KWARGS = {'n_actions': N_ACTIONS, 'n_items': N_ACTIONS}   # ys carries the target mask as a 3rd column


class SpiceModel(BaseModel):
    """Six stateful modules, one per act group of ID1 and ID2, read out as P(ID2 grooms next)."""

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

            self.call_module(
                key_module='negotiation_observed',
                key_state='value_negotiation_observed',
                action_mask=mask_groom,
                inputs=(
                    signals['id1_grooming_solicitation'][trial],
                    signals['id1_self_reposition'][trial],
                    signals['id1_directed_scratch'][trial],
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='grooming_preparation_received',
                key_state='value_grooming_preparation_received',
                action_mask=mask_groom,
                inputs=(
                    signals['id1_reposition_body'][trial],
                    signals['id1_grooming_process'][trial],
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='grooming_received',
                key_state='value_grooming_received',
                action_mask=mask_groom,
                inputs=(
                    signals['id1_groom'][trial],
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='negotiation_given',
                key_state='value_negotiation_given',
                action_mask=mask_groom,
                inputs=(
                    signals['id2_grooming_solicitation'][trial],
                    signals['id2_self_reposition'][trial],
                    signals['id2_directed_scratch'][trial],
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='grooming_preparation_given',
                key_state='value_grooming_preparation_given',
                action_mask=mask_groom,
                inputs=(
                    signals['id2_reposition_body'][trial],
                    signals['id2_grooming_process'][trial],
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='grooming_given',
                key_state='value_grooming_given',
                action_mask=mask_groom,
                inputs=(
                    signals['id2_groom'][trial],
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            spice_signals.logits[trial] = \
                + self.state['value_negotiation_observed'] \
                + self.state['value_grooming_preparation_received'] \
                + self.state['value_grooming_received'] \
                + self.state['value_negotiation_given'] \
                + self.state['value_grooming_preparation_given'] \
                + self.state['value_grooming_given']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
