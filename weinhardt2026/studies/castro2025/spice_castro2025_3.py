import torch

from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        # Reward learning: chosen item updates from reward + cross-arm mean
        # SINDy can capture: Rescorla-Wagner, loss aversion (linear in reward),
        # exploration smoothing toward mean, and decay (via self-state coeff)
        'value_reward_chosen': [
            'reward[t]',
            'value_reward_mean',
        ],
        # Unchosen items: decay toward cross-arm mean (exploration smoothing + forgetting)
        'value_reward_not_chosen': [
            'value_reward_mean',
        ],
        # Choice persistence for chosen item: action[t-1] distinguishes
        # perseveration (action[t-1]=1) from switching (action[t-1]=0),
        # enabling SINDy to learn separate update rules for each case
        'value_choice_chosen': [
            'action[t-1]',
        ],
        # Choice persistence for unchosen items: action[t-1] distinguishes
        # recently-abandoned items (action[t-1]=1) from already-unchosen (action[t-1]=0)
        'value_choice_not_chosen': [
            'action[t-1]',
        ],
        # Exploration driven by reward value changes (captures volatility-driven
        # exploration dynamics that the benchmark approximates via decaying exploration rate)
        'value_exploration_chosen': [
            'dvalue',
        ],
        'value_exploration_not_chosen': [
        ],
    },
    memory_state={
        'value_reward': 0.,
        'value_choice': 0.,
        'value_exploration': 0.,

        # Buffers (excluded from logits)
        'value_reward[t-1]': 0.,
        'action[t-1]': 0.,
    },
    states_in_logit=['value_reward', 'value_choice', 'value_exploration'],
)


class SpiceModel(BaseModel):
    """
    SPICE model for drifting multi-armed bandit (Castro et al. 2025 / Eckstein 2024).

    Three cognitive mechanisms decomposed into polynomial-amenable submodules:

    1. Reward learning (value_reward):
       - Chosen: learns from reward and cross-arm mean value. The mean input
         enables exploration smoothing (Q -> mean(Q) reversion) within SINDy.
       - Unchosen: mean input enables decay toward cross-arm average.

    2. Choice persistence (value_choice):
       - Key improvement over v2: receives action[t-1] as per-item input.
         For the chosen item, action[t-1]=1 if perseverating (same arm),
         action[t-1]=0 if switching (new arm). This lets SINDy learn distinct
         perseveration vs. switching dynamics via the interaction term
         state*action[t-1], which the benchmark captures through explicit
         perseveration_strength and switch_strength parameters.

    3. Exploration (value_exploration):
       - Tracks reward volatility via dvalue = value_reward[t] - value_reward[t-1].
         In a drifting bandit, large reward changes signal arm instability and
         should drive exploration. The benchmark approximates this with a fixed
         exponentially-decaying exploration rate.

    Design changes vs. v2:
      - Dropped value_reward_env: the mean_value_reward signal already provides
        environmental context; removing the env module reduces SINDy complexity.
      - Replaced mean_value_choice with action[t-1] in choice modules: provides
        the critical perseveration/switching signal that was missing.
      - Renamed volatility -> exploration: better reflects the functional role.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        # Reward modules: 2 and 1 control signals
        self.setup_module(key_module='value_reward_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=1, dropout=self.dropout)

        # Choice modules: 1 control signal each (action[t-1])
        self.setup_module(key_module='value_choice_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=1, dropout=self.dropout)

        # Exploration modules: 1 control signal each (dvalue)
        self.setup_module(key_module='value_exploration_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_not_chosen', input_size=0, dropout=self.dropout)

    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- REWARD VALUE UPDATES ---
            # Cross-arm mean enables exploration smoothing: the module can learn
            # state += c*(mean - state), mirroring the benchmark's
            # Q = (1-er)*Q + er*mean(Q)
            mean_value_reward = self.state['value_reward'].mean(
                dim=-1, keepdim=True,
            ).expand_as(self.state['value_reward']).detach()

            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.rewards[trial],
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
                    mean_value_reward,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- CHOICE VALUE UPDATES ---
            # action[t-1] is per-item: 1 if that item was chosen last trial, 0 otherwise.
            # Combined with action_mask, the chosen module sees:
            #   action[t-1]=1 when perseverating (same arm as last trial)
            #   action[t-1]=0 when switching (different arm)
            # SINDy can then learn: state += a + b*state + c*action[t-1] + d*state*action[t-1]
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['action[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    self.state['action[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- EXPLORATION VALUE UPDATES ---
            # dvalue captures per-item reward change magnitude.
            # Detached to prevent cross-module SINDy gradient interference.
            dvalue = (self.state['value_reward'] - self.state['value_reward[t-1]']).detach()

            self.call_module(
                key_module='value_exploration_chosen',
                key_state='value_exploration',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    dvalue,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_exploration_not_chosen',
                key_state='value_exploration',
                action_mask=1 - spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- BUFFER UPDATES ---
            self.state['value_reward[t-1]'] = self.state['value_reward']
            self.state['action[t-1]'] = spice_signals.actions[trial]

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_reward']
                + self.state['value_choice']
                + self.state['value_exploration']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
