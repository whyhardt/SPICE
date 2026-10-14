import torch

from spice import SpiceConfig, BaseModel


# -------------------------------------------------------------------------------------------
# TOBETO DRIVE MODEL
# -------------------------------------------------------------------------------------------
# Predicts the focal ape's own next act, grouped by the grooming role it is after
# ("TO BE groomed" vs "TO groom", see data/PREPROCESSING.md for the category mapping).
#
# Readout (4 items, one softmax):
#   0 none         -- the focal ape does not act next (the partner does); reference class
#   1 tobegroomed  -- Grooming_Solicitation, Self_Reposition, Directed_Scratch
#   2 togroom      -- Reposition_Body, Grooming_Process
#   3 groom
#
# Each written item is a latent drive, updated after every event from what the focal ape
# and its partner just did:
#   drive_tobegroomed     -- drive to be groomed, shown by to-be-groomed signals
#   drive_togroom_signal  -- drive to groom, shown by to-groom signals
#   drive_togroom_groom   -- drive to groom, shown by grooming itself
# The drive to groom is split into two modules because grooming persists over consecutive
# events while signals do not; a shared equation would fix their ratio.
#
# The library is linear (degree 1): per module {1, drive, 6 act indicators, rank_diff}.
# The dominance distance to the partner is constant within an interaction, so its term acts
# as a dyad-specific shift of the drive's resting level.
# -------------------------------------------------------------------------------------------

N_ITEMS = 4
ITEM_NONE, ITEM_TOBEGROOMED, ITEM_TOGROOM, ITEM_GROOM = range(N_ITEMS)
ITEM_NAMES = ('none', 'tobegroomed', 'togroom', 'groom')

# Which dominance-distance signal feeds the `rank_diff` library term.
#   'rank_diff_centered' -- deviation from the ape's typical partner; orthogonal to its own
#                           rank, so a surviving term means the ape adjusts to *this* partner.
#   'rank_diff'          -- raw difference; ~collinear with own rank and hence with the bias.
RANK_SIGNAL = 'rank_diff_centered'

SIGNALS = (
    'own_tobegroomed', 'own_togroom', 'own_groom',
    'partner_tobegroomed', 'partner_togroom', 'partner_groom',
    'rank_diff',
)

# module -> readout item it writes
MODULE_ITEMS = {
    'drive_tobegroomed': ITEM_TOBEGROOMED,
    'drive_togroom_signal': ITEM_TOGROOM,
    'drive_togroom_groom': ITEM_GROOM,
}

CONFIG = SpiceConfig(
    library_setup={module: list(SIGNALS) for module in MODULE_ITEMS},
    memory_state={
        'drive': None,  # [none, tobegroomed, togroom, groom]; per-ape learnable initial value
    },
    states_in_logit=['drive'],
    additional_inputs=('own_act', 'partner_act', 'rank_own', 'rank_partner',
                       'rank_diff', 'rank_diff_centered'),
)


class SpiceModel(BaseModel):
    """Three latent grooming drives, read out as the focal ape's next TOBETO act."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

    def _act_indicators(self, act_codes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Item codes -> [tobegroomed, togroom, groom] indicators, padded trials zeroed.

        `init_forward_pass` replaces NaN by 0, which is also the code for "no act"; the
        valid-trial mask keeps padding from being read as anything else. "No act" is the
        all-zero reference level and gets no indicator.
        """
        indicators = torch.eye(N_ITEMS, device=self.device)[act_codes.squeeze(-1).int()]
        return indicators[..., ITEM_NONE + 1:] * valid

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # [T, E, B, 1] -> [T, 1, E, B, 1]: valid across the (singleton) within-trial axis
        valid = spice_signals.mask_valid_trials.unsqueeze(1).float()

        indicators = torch.cat((
            self._act_indicators(spice_signals.additional_inputs['own_act'], valid),
            self._act_indicators(spice_signals.additional_inputs['partner_act'], valid),
            spice_signals.additional_inputs[RANK_SIGNAL] * valid,
        ), dim=-1)  # [T, W, E, B, len(SIGNALS)]

        # each module writes exactly one item; `none` is never written (softmax reference)
        masks = {}
        for module, item in MODULE_ITEMS.items():
            masks[module] = torch.zeros_like(spice_signals.actions[0])
            masks[module][..., item] = 1

        for timestep in spice_signals.trials:

            inputs_t = tuple(
                indicators[timestep, ..., index:index + 1] for index in range(len(SIGNALS))
            )

            for module in MODULE_ITEMS:
                self.call_module(
                    key_module=module,
                    key_state='drive',
                    action_mask=masks[module],
                    inputs=inputs_t,
                    participant_index=spice_signals.participant_ids,
                    participant_embedding=participant_embedding,
                )

            spice_signals.logits[timestep] = self.state['drive']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()


def filter_own_acts(ys: torch.Tensor) -> torch.Tensor:
    """Return a (B, T) bool mask that is True where the focal ape itself acts next."""
    return torch.argmax(ys[:, :, 0, :], dim=-1) != ITEM_NONE
