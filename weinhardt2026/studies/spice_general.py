import torch

from spice import BaseModel
from spice import SpiceConfig


"""
General-purpose SPICE architecture.

Generalizes the eckstein2026 model (spice_eckstein2026.py) — currently SPICE's
strongest fit, outperforming an LLM-derived cognitive model on a four-armed
bandit — by dropping its one genuinely task-specific piece: `bias_attention`,
which only makes sense when actions have a spatial/circular layout (adjacent
vs. opposite arms).

Everything else here is architecture-agnostic with respect to what "actions"
represent:
  - Item-identity tasks (n-armed bandits): actions index distinct value
    representations (eckstein2026, ganesh2024a, dezfouli2019-style).
  - Fixed binary action frames (repeat/switch, harvest/exit): actions index a
    task-structural choice rather than an item, but the chosen/unchosen split
    is exactly the same one-hot masking either way (bustamante2023, braun2018).

So this model adapts across studies purely through SpiceEstimator's
n_actions/n_items and this file's CONFIG.additional_inputs — no forward()
changes needed for that class of task differences.

For task-specific mechanisms beyond this core (spatial bias, fatigue/block
effects, patch depletion, perceptual certainty, ...), follow the existing
per-study convention: copy this file into the study directory as
spice_<study>.py and extend CONFIG/forward() there, rather than subclassing —
forward() here uses local variables (reward, dvalue, participant_embedding)
that aren't exposed as override hooks.
"""


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_env': (
            'reward[t]',
        ),
        'value_reward_chosen': (
            'reward_env',
            'reward[t]',
            'value_reward_mean',
        ),
        'value_reward_not_chosen': (
            'reward_env',
            'value_reward_mean',
        ),
        'value_choice': (
            'action[t]',
            'action[t-1]',
        ),
        'value_exploration_chosen': (
            'dvalue_pos',
            'dvalue_neg',
        ),
        'value_exploration_not_chosen': (
            'dvalue_pos',
            'dvalue_neg',
        ),
    },
    memory_state={
        'value_reward_env': None,
        'value_reward': None,
        'value_choice': None,
        'value_exploration': None,

        # Buffers (excluded from logits)
        'value_reward[t-1]': None,
        'action[t-1]': 0,
    },
    states_in_logit=[
        'value_reward_env',
        'value_reward',
        'value_choice',
        'value_exploration',
    ],
)


class SpiceModel(BaseModel):
    """
    Core reward-learning + choice-perseveration + exploration architecture,
    generalized across item-identity and fixed-binary-action-frame tasks.

    5 modules feeding 4 logit states. See module docstring for provenance
    and extension pattern.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        self.setup_module(key_module='value_reward_env', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_reward_chosen', input_size=3, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_choice', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_not_chosen', input_size=2, dropout=self.dropout)

    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        reward_full = spice_signals.feedback.sum(dim=-1, keepdim=True).expand_as(spice_signals.actions)
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- ENV REWARD ---
            value_reward_env = self.call_module(
                key_module='value_reward_env',
                key_state='value_reward_env',
                inputs=reward_full[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- REWARD VALUE UPDATES ---
            mean_value_reward = self.state['value_reward'].mean(
                dim=-1, keepdim=True,
            ).expand_as(self.state['value_reward']).detach()

            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    spice_signals.feedback[trial],
                    mean_value_reward,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    mean_value_reward,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- CHOICE VALUE UPDATES (perseveration) ---
            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                inputs=(
                    spice_signals.actions[trial],
                    self.state['action[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- EXPLORATION VALUE UPDATES ---
            dvalue = (self.state['value_reward'] - self.state['value_reward[t-1]']).detach()
            dvalue_pos = torch.relu(dvalue)
            dvalue_neg = torch.relu(-dvalue)

            self.call_module(
                key_module='value_exploration_chosen',
                key_state='value_exploration',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    dvalue_pos,
                    dvalue_neg,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_exploration_not_chosen',
                key_state='value_exploration',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    dvalue_pos,
                    dvalue_neg,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- BUFFER UPDATES ---
            self.state['value_reward[t-1]'] = self.state['value_reward']
            self.state['action[t-1]'] = spice_signals.actions[trial]

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_reward_env']
                + self.state['value_reward']
                + self.state['value_choice']
                + self.state['value_exploration']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
