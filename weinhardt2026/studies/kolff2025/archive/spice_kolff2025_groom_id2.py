import torch

from spice import SpiceConfig, BaseModel, SpiceDataset


# -------------------------------------------------------------------------------------------
# ID2 GROOMING MODEL
# -------------------------------------------------------------------------------------------
# ID2 (approached ape) is the participant. Predicts whether ID2's next act is grooming
# (item 1) or not (item 0, reference logit 0); scored only on events where ID2 acts next.
#
# Three stateless modules see the same drive_to_groom (passed as an input) together with one
# act group of ID1 and the rank difference. Each returns its update to the drive; the updates
# add up: drive_{t+1} = drive_t + sum of updates (= grooming logit).
# -------------------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        'negotiation_observed': [
            'drive_to_groom', 
            'negotiation', 
            'rank_difference',
            ],
        'grooming_preparation_received': [
            'drive_to_groom', 
            'preparation', 
            'rank_difference',
            ],
        'grooming_received': [
            'drive_to_groom',
            'grooming',
            'rank_difference',
            ],
        'negotiation_given': [
            'drive_to_groom',
            'own_negotiation',
            'rank_difference',
            ],
        'grooming_preparation_given': [
            'drive_to_groom',
            'own_preparation',
            'rank_difference',
            ],
        'grooming_given': [
            'drive_to_groom',
            'own_grooming',
            'rank_difference',
            ],
        },
    memory_state={
        'drive_to_groom': 0,
        'negotiation_update': 0,
        'grooming_preparation_update': 0,
        'grooming_update': 0,
        'own_negotiation_update': 0,
        'own_grooming_preparation_update': 0,
        'own_grooming_update': 0,
    },
    states_in_logit=[
        # 'drive_to_groomed',
        'negotiation_update',
        'grooming_preparation_update',
        'grooming_update',
        'own_negotiation_update',
        'own_grooming_preparation_update',
        'own_grooming_update',
        ],
    additional_inputs=(
        'id1_negotiation',
        'id1_prepare',
        'id1_groom',
        'id2_negotiation',
        'id2_prepare',
        'id2_groom',
        'rank_diff_centered',
        'id2_acts_next',  # target mask only, never a model input (future information)
        ),
)

BINARY_SIGNALS = ('negotiation', 'preparation', 'grooming', 'own_negotiation', 'own_preparation', 'own_grooming')

N_ACTIONS = 2  # [not groom, groom]
COL_TARGET_MASK = N_ACTIONS + CONFIG.additional_inputs.index('id2_acts_next')


def target_mask(ys: torch.Tensor) -> torch.Tensor:
    """True where ID2 acts at the next event, i.e. where a target exists. ys (..., 3) -> (...)."""
    return (ys[..., N_ACTIONS] == 1) & ~torch.isnan(ys[..., 0])


def masked_cross_entropy(prediction: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0., **kwargs) -> torch.Tensor:
    """Cross-entropy over the events where ID2 acts next; target = [not groom, groom, mask]."""
    prediction = prediction.reshape(-1, N_ACTIONS)
    target = target.reshape(-1, target.shape[-1])
    mask = target_mask(target)
    if not mask.any():
        return prediction.sum() * 0.
    return torch.nn.functional.cross_entropy(
        prediction[mask], target[mask, :N_ACTIONS].argmax(-1), label_smoothing=label_smoothing,
    )


def prepare_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """Pack the target mask into ys: ys = [not groom, groom, ID2 acts next].

    The choice column is 0/1 on every event, so events where only ID1 acts stay valid input
    steps; which of them are scored is decided by the mask alone.
    """
    mask = dataset.xs[..., COL_TARGET_MASK:COL_TARGET_MASK + 1].clone()
    return SpiceDataset(dataset.xs, torch.cat((dataset.ys, mask), dim=-1), n_reward_features=0)


# Hooks read by weinhardt2026/run.py
NORMALIZE_REWARDS = False       # no reward columns
LOSS_FN = masked_cross_entropy
LOSS_FN_KWARGS = {'label_smoothing': 0.01}
ESTIMATOR_KWARGS = {'n_actions': N_ACTIONS, 'n_items': N_ACTIONS}   # ys carries the target mask as a 3rd column


class SpiceModel(BaseModel):
    """ID2's drive to groom, updated by what ID1 does and read out as P(ID2 grooms next)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

        self.setup_module('negotiation_observed', include_state=False)
        self.setup_module('grooming_preparation_received', include_state=False)
        self.setup_module('grooming_received', include_state=False)
        self.setup_module('negotiation_given', include_state=False)
        self.setup_module('grooming_preparation_given', include_state=False)
        self.setup_module('grooming_given', include_state=False)

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

            negotiation = spice_signals.additional_inputs['id1_negotiation'][trial]
            preparation = spice_signals.additional_inputs['id1_prepare'][trial]
            grooming = spice_signals.additional_inputs['id1_groom'][trial]
            own_negotiation = spice_signals.additional_inputs['id2_negotiation'][trial]
            own_preparation = spice_signals.additional_inputs['id2_prepare'][trial]
            own_grooming = spice_signals.additional_inputs['id2_groom'][trial]
            rank_difference = spice_signals.additional_inputs['rank_diff_centered'][trial]

            update_negotiation = self.call_module(
                key_module='negotiation_observed',
                key_state='negotiation_update',
                action_mask=mask_groom,
                inputs=(
                    self.state['drive_to_groom'], 
                    negotiation, 
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            update_preparation = self.call_module(
                key_module='grooming_preparation_received',
                key_state='grooming_preparation_update',
                action_mask=mask_groom,
                inputs=(
                    self.state['drive_to_groom'], 
                    preparation, 
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            update_grooming = self.call_module(
                key_module='grooming_received',
                key_state='grooming_update',
                action_mask=mask_groom,
                inputs=(
                    self.state['drive_to_groom'], 
                    grooming, 
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            update_own_negotiation = self.call_module(
                key_module='negotiation_given',
                key_state='own_negotiation_update',
                action_mask=mask_groom,
                inputs=(
                    self.state['drive_to_groom'],
                    own_negotiation,
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            update_own_preparation = self.call_module(
                key_module='grooming_preparation_given',
                key_state='own_grooming_preparation_update',
                action_mask=mask_groom,
                inputs=(
                    self.state['drive_to_groom'],
                    own_preparation,
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            update_own_grooming = self.call_module(
                key_module='grooming_given',
                key_state='own_grooming_update',
                action_mask=mask_groom,
                inputs=(
                    self.state['drive_to_groom'],
                    own_grooming,
                    rank_difference,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.state['drive_to_groom'] = self.state['drive_to_groom'] \
                + update_negotiation \
                + update_preparation \
                + update_grooming \
                + update_own_negotiation \
                + update_own_preparation \
                + update_own_grooming
            
            spice_signals.logits[trial] = self.state['drive_to_groom']    
            # spice_signals.logits[trial] = \
                # + self.state['negotiation_update'] \
                # + self.state['grooming_preparation_update'] \
                # + self.state['grooming_update']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
