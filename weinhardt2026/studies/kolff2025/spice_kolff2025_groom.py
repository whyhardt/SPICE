import torch

from spice import SpiceConfig, BaseModel


# -------------------------------------------------------------------------------------------
# GROOMING-EXPECTATION MODEL
# -------------------------------------------------------------------------------------------
# Rather than predicting which of five signalling acts the focal ape emits next — a 5-way
# choice that is too noisy for the number of events available — this model tracks how
# strongly the focal ape expects grooming to happen next, and in which direction.
#
# Readout (3 items, one softmax):
#   0 negotiate     — the exchange continues without grooming (the reference class)
#   1 owngroom      — the focal ape grooms its partner next
#   2 partnergroom  — the focal ape gets groomed next
#
# The expectation is decomposed additively into two channels, by what drives it:
#   value_ownact     — what the focal ape's own acts imply about what comes next
#   value_partneract — what the received acts imply about what comes next
# Own and partner acts never co-occur in these data (0.1% overlap), so the two channels are
# driven on disjoint, alternating subsets of events and idle in pure decay otherwise. The
# split therefore earns its parameters only if the two decay differently — which is exactly
# the hypothesis it is there to test.
#
# A third state carries the effect of dominance distance on the baseline propensity to
#
# Waiting has no indicator of its own: in the one-hot encoding it is the absence of every
# other act, so it is the reference level and every coefficient reads "relative to doing
# nothing".
# -------------------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        # what the focal ape's own act implies for each grooming direction
        'ownact_owngroom': ['own_acts', 'own_grooms', 'own_gestures', 'own_scratches', 'rank_diff'],
        'ownact_partnergroom': ['own_acts', 'own_grooms', 'own_gestures', 'own_scratches', 'rank_diff'],
        # what the received act implies for each grooming direction
        'partneract_owngroom': ['partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches', 'rank_diff'],
        'partneract_partnergroom': ['partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches', 'rank_diff'],
    },
    memory_state={
        # feature dimensions of every state are [negotiate, owngroom, partnergroom]
        'value_ownact': None,      # per-participant learnable initial value
        'value_partneract': 0.,    # pinned: only the *sum* of the initial values is identified
    },
    states_in_logit=['value_ownact', 'value_partneract'],
    additional_inputs=('own_action', 'partner_action', 'rank_own', 'rank_partner',
                       'rank_diff', 'rank_diff_centered'),
)

# Which dominance-distance signal feeds the `rank_diff` library term.
#   'rank_diff_centered' -- within-ape deviation; orthogonal to rank_own by construction,
#                           so its coefficient can be regressed against rank without
#                           circularity. Answers "how does this ape adjust across its own
#                           partners".
#   'rank_diff'          -- raw difference. Retains 37% more spread but is ~collinear with
#                           rank_own (rho = 0.92 at the ape level).
RANK_SIGNAL = 'rank_diff_centered'

N_SIGNAL_ACTS = 5  # action, grooming, gesture, scratching, waiting -- the control alphabet
ITEM_NEGOTIATE, ITEM_OWNGROOM, ITEM_PARTNERGROOM = 0, 1, 2

# Own and partner acts never co-occur, so all eight indicators form a *single* mutually
# exclusive group: x^2 = x and x_i * x_j = 0 for any pair, across blocks as well as within.
EXCLUSIVE_GROUPS = [(
    'own_acts', 'own_grooms', 'own_gestures', 'own_scratches',
    'partner_acts', 'partner_grooms', 'partner_gestures', 'partner_scratches',
)]
BINARY_SIGNALS = {signal for group in EXCLUSIVE_GROUPS for signal in group}


class SpiceModel(BaseModel):
    """Three-way grooming-expectation model with own-act / received-act decomposition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for one-hot indicators.

        For binary indicators x^2 = x, and for two indicators of the same mutually
        exclusive group x_i * x_j = 0. Removing them up front leaves the gated-affine
        terms that can actually be identified: {1, V, V^2, x_i, V * x_i}. The continuous
        rank signal is not filtered.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            for index_term, term in enumerate(candidate_terms[module]):
                redundant = any(
                    signal + '^' in term for signal in control_signals if signal in BINARY_SIGNALS
                )
                for group in EXCLUSIVE_GROUPS:
                    group_signals = [signal for signal in control_signals if signal in group]
                    for index_signal, signal_i in enumerate(group_signals):
                        for signal_j in group_signals[index_signal + 1:]:
                            redundant |= (signal_i in term and signal_j in term)
                if redundant:
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

    def _act_indicators(self, act_codes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Scalar act codes -> per-act indicators, with padded trials zeroed out.

        `init_forward_pass` replaces NaN by 0, which would otherwise encode padding as
        "the ape performed act 0"; multiplying by the valid-trial mask removes that.
        Waiting (the last act) is dropped -- it is the all-zero reference level.

        The signalling-act alphabet (5) is not the readout alphabet (3), so the indicators
        cannot be read off `spice_signals.actions` and are one-hot encoded here from the
        additional inputs instead.
        """
        indicators = torch.eye(N_SIGNAL_ACTS, device=self.device)[act_codes.squeeze(-1).int()]
        return indicators[..., :N_SIGNAL_ACTS - 1] * valid

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # [T, E, B, 1] -> [T, 1, E, B, 1]: valid across the (singleton) within-trial axis
        valid = spice_signals.mask_valid_trials.unsqueeze(1).float()

        own_indicators = self._act_indicators(spice_signals.additional_inputs['own_action'], valid)
        partner_indicators = self._act_indicators(spice_signals.additional_inputs['partner_action'], valid)

        # Dominance distance, precomputed and linear (see RANK_SIGNAL above for which
        # variant). Ranks are normalised within community -- the two hierarchies differ in
        # depth -- so the difference is on a common scale and antisymmetric by
        # construction. With no learned transform in the way, the SINDy coefficient on this
        # term reads directly as an effect per unit of normalised rank distance.
        rank_diff = spice_signals.additional_inputs[RANK_SIGNAL] * valid  # [T, W, E, B, 1]
        
        # item masks: each module writes exactly one grooming direction; `negotiate` is
        # never written and stays at its initial value, serving as the softmax reference
        mask_owngroom = torch.zeros_like(spice_signals.actions[0])
        mask_owngroom[..., ITEM_OWNGROOM] = 1
        mask_partnergroom = torch.zeros_like(spice_signals.actions[0])
        mask_partnergroom[..., ITEM_PARTNERGROOM] = 1

        for timestep in spice_signals.trials:

            ownact = tuple(list(
                own_indicators[timestep, ..., index:index + 1] for index in range(N_SIGNAL_ACTS - 1)
            ) + [rank_diff[timestep]])
            partneract = tuple(list(
                partner_indicators[timestep, ..., index:index + 1] for index in range(N_SIGNAL_ACTS - 1)
            ) + [rank_diff[timestep]])

            # --- what my own act implies ---
            self.call_module(
                key_module='ownact_owngroom',
                key_state='value_ownact',
                action_mask=mask_owngroom,
                inputs=ownact,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='ownact_partnergroom',
                key_state='value_ownact',
                action_mask=mask_partnergroom,
                inputs=ownact,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- what the received act implies ---
            self.call_module(
                key_module='partneract_owngroom',
                key_state='value_partneract',
                action_mask=mask_owngroom,
                inputs=partneract,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='partneract_partnergroom',
                key_state='value_partneract',
                action_mask=mask_partnergroom,
                inputs=partneract,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- logits: additive composition of the two channels ---
            spice_signals.logits[timestep] = (
                self.state['value_ownact']
                + self.state['value_partneract']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()


def filter_grooming(ys: torch.Tensor) -> torch.Tensor:
    """Return a (B, T) bool mask that is True where grooming happens next in either direction."""
    return torch.argmax(ys[:, :, 0, :], dim=-1) != ITEM_NEGOTIATE
