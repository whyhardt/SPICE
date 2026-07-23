"""
Joint choice + response-time model for eckstein2026.

Belief part: the simple RL architecture from spice/precoded/choice.py
(reward-based values + choice perseveration).

RT part: a DDM-style decision module informed by those beliefs, fit via an
exact discrete-time hazard/survival likelihood (as in RTify's example2.ipynb)
instead of simulating noisy evidence trajectories and detecting threshold
crossings (spice/precoded/ddm.py). The evidence accumulation is deterministic
and the stopping-time distribution is computed once, in closed form, for all
trials at once -- fully differentiable, no custom autograd Function needed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator, BaseModel, SpiceConfig, SpiceDataset, csv_to_dataset, split_data_along_blockdim


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
        'value_choice': ['choice'],
        'drift': ['value_gap'],
    },
    memory_state={
        'value_reward': 0.,
        'value_choice': 0.,
        'drift': 0.,
    },
    states_in_logit=['value_reward', 'value_choice'],
)


class SpiceModel(BaseModel):
    """RL beliefs (spice/precoded/choice.py) + a hazard-based RT decision module.

    Each trial, the belief modules produce action values as usual. The gap
    between the best and second-best value (a proxy for decision difficulty)
    drives a learned 'drift' module. After the belief loop, the drift for
    every trial is turned into a discrete-time stopping-time distribution
    over `max_steps` bins of width `dt = t_max / max_steps`:

        evidence[w] = drift * (w+1) * dt                     (deterministic)
        p_stop[w]   = sigmoid(gain * (|evidence[w]| - threshold))
        p_decision[w] = p_stop[w] * prod_{k<w}(1 - p_stop[k])

    `p_decision` is a proper probability distribution over RT bins (leftover
    mass is absorbed into the last bin), so RTs can be fit by direct
    negative log-likelihood instead of distribution matching.
    """

    def __init__(self, max_steps: int = 40, t_max: float = 2.0, **kwargs):
        super().__init__(**kwargs)

        self.max_steps = max_steps
        self.t_max = t_max
        self.dt = t_max / max_steps

        self.hazard_gain = torch.nn.Parameter(torch.tensor(0.))
        self.hazard_threshold = torch.nn.Parameter(torch.tensor(0.))

        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=self.dropout)

        self.setup_module(key_module='value_reward_chosen', input_size=1, embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0, embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='value_choice', input_size=1, embedding_size=self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='drift', input_size=1, embedding_size=self.embedding_size, dropout=self.dropout)

    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None):
        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        drift_trials = []

        for timestep in spice_signals.trials:

            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.feedback[timestep],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                action_mask=None,
                inputs=spice_signals.actions[timestep],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            values_trial = self.state['value_reward'] + self.state['value_choice']  # [1, E, B, n_actions]
            top2 = torch.topk(values_trial, k=2, dim=-1).values
            value_gap = (top2[..., 0:1] - top2[..., 1:2])  # [1, E, B, 1]

            self.call_module(
                key_module='drift',
                key_state='drift',
                action_mask=None,
                inputs=value_gap,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            drift_trials.append(self.state['drift'][..., 0:1])  # [1, E, B, 1]
            spice_signals.logits[timestep] = values_trial

        spice_signals = self.post_forward_pass(spice_signals)  # logits: [E, B, T, 1, n_actions]

        drift = torch.stack(drift_trials, dim=0)  # [T, 1, E, B, 1]
        drift = drift.permute(2, 3, 0, 1, 4)  # [E, B, T, 1, 1]

        gain = torch.nn.functional.softplus(self.hazard_gain)
        threshold = torch.nn.functional.softplus(self.hazard_threshold)

        steps = torch.arange(1, self.max_steps + 1, device=self.device, dtype=drift.dtype)  # [max_steps]
        evidence = torch.abs(drift) * steps * self.dt  # [E, B, T, 1, max_steps]

        p_stop = torch.sigmoid(gain * (evidence - threshold))
        survival = torch.cumprod(
            torch.cat((torch.ones_like(p_stop[..., :1]), 1. - p_stop[..., :-1]), dim=-1),
            dim=-1,
        )
        p_decision = p_stop * survival  # [E, B, T, 1, max_steps]
        remainder = (1. - p_decision.sum(dim=-1, keepdim=True)).clamp(min=0.)
        p_decision = torch.cat((p_decision[..., :-1], p_decision[..., -1:] + remainder), dim=-1)

        output = torch.cat((spice_signals.logits, p_decision), dim=-1)  # [E, B, T, 1, n_actions + max_steps]

        return output, self.get_state()


def make_rt_loss(n_actions: int, dt: float, max_steps: int, rt_weight: float = 1.0):
    """Joint negative log-likelihood of choice and (discretized) response time.

    `prediction` packs [choice_logits (n_actions), p_decision (max_steps)].
    `target` packs [choice_onehot (n_actions), rt_seconds (1)].
    """

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        choice_logits = prediction[..., :n_actions]
        p_decision = prediction[..., n_actions:]
        choice_target = target[..., :n_actions]
        rt_target = target[..., n_actions]

        choice_idx = torch.argmax(choice_target, dim=-1)
        loss_choice = torch.nn.functional.cross_entropy(choice_logits, choice_idx)

        rt_valid = ~torch.isnan(rt_target)
        rt_bin = torch.clamp((rt_target[rt_valid] / dt).long(), 0, max_steps - 1)
        p_obs = p_decision[rt_valid].gather(-1, rt_bin.unsqueeze(-1)).squeeze(-1)
        loss_rt = -torch.log(p_obs.clamp_min(1e-8)).mean()

        return loss_choice + rt_weight * loss_rt

    return loss_fn


def get_dataset(path_data: str, test_blocks: tuple = None, verbose: bool = False):
    """Load eckstein2026 choices+RTs into a SpiceDataset with joint targets.

    `additional_inputs=['rt'], timeshift_additional_inputs=[-1]` pulls each
    trial's *next* RT into the xs column for 'rt' (mirroring how ys already
    holds the next action). That column is then moved out of xs (so the
    model never sees it as an input) and appended to ys in seconds, giving
    ys = [next_action_onehot, next_rt_seconds].
    """
    raw = csv_to_dataset(
        file=path_data,
        additional_inputs=['rt'],
        timeshift_additional_inputs=[-1],
    )

    n_actions = raw.n_actions
    rt_col = n_actions + raw.n_reward_features

    rt_next_ms = raw.xs[..., rt_col].clone()
    rt_next_ms[:, -1] = float('nan')  # last slot has no "next" trial to draw from

    xs = torch.cat((raw.xs[..., :rt_col], raw.xs[..., rt_col + 1:]), dim=-1)
    ys = torch.cat((raw.ys, (rt_next_ms / 1000.).unsqueeze(-1)), dim=-1)

    dataset = SpiceDataset(xs, ys, n_reward_features=raw.n_reward_features)
    dataset.normalize_rewards()

    if verbose:
        print(f"Shape of dataset: {dataset.xs.shape}")
        print(f"Number of participants: {dataset.n_participants}")
        print(f"Number of actions: {n_actions}")

    if test_blocks is not None:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train, dataset_test = dataset, None

    info_dataset = {'n_participants': dataset.n_participants, 'n_actions': n_actions}
    return dataset_train, dataset_test, info_dataset


def subset_dataset(dataset: SpiceDataset, n_participants: int = None, n_trials: int = None) -> SpiceDataset:
    """Restrict a dataset to the first `n_participants` participants and first `n_trials` trials.

    Useful for fast prototyping iterations before committing to a full run.
    """
    xs, ys = dataset.xs, dataset.ys

    if n_participants is not None:
        session_mask = xs[:, 0, 0, -1] < n_participants
        xs, ys = xs[session_mask], ys[session_mask]

    if n_trials is not None:
        xs, ys = xs[:, :n_trials], ys[:, :n_trials]

    return SpiceDataset(xs, ys, n_reward_features=dataset.n_reward_features)


@torch.no_grad()
def evaluate(estimator: SpiceEstimator, dataset: SpiceDataset, n_actions: int, dt: float, max_steps: int) -> dict:
    """Choice accuracy and mean predicted RT vs. observed RT."""
    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)

    prediction, _ = estimator.model(xs)
    prediction = prediction.mean(dim=0)  # average over ensemble: [B, T, 1, n_actions + max_steps]
    mask = ~torch.isnan(xs[..., :n_actions].sum(dim=-1))

    choice_logits = prediction[..., :n_actions][mask]
    p_decision = prediction[..., n_actions:][mask]
    choice_target = ys[..., :n_actions][mask]
    rt_target = ys[..., n_actions][mask]

    accuracy = (choice_logits.argmax(dim=-1) == choice_target.argmax(dim=-1)).float().mean().item()

    bins = torch.arange(max_steps, device=p_decision.device, dtype=p_decision.dtype) * dt + dt / 2
    rt_pred_mean = (p_decision * bins).sum(dim=-1).mean().item()

    rt_valid = ~torch.isnan(rt_target)
    rt_obs_mean = rt_target[rt_valid].mean().item()

    return {'choice_accuracy': accuracy, 'rt_pred_mean': rt_pred_mean, 'rt_obs_mean': rt_obs_mean}


@torch.no_grad()
def plot_rt_distribution(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    n_actions: int,
    dt: float,
    max_steps: int,
    output_path: str = None,
):
    """Histogram of observed RTs vs. RTs sampled from the model's decision-time distribution."""
    import matplotlib.pyplot as plt
    import numpy as np

    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)

    prediction, _ = estimator.model(xs)
    prediction = prediction.mean(dim=0)
    mask = ~torch.isnan(xs[..., :n_actions].sum(dim=-1))

    p_decision = prediction[..., n_actions:][mask]
    rt_target = ys[..., n_actions][mask]

    rt_valid = ~torch.isnan(rt_target)
    p_decision = p_decision[rt_valid]
    rt_obs = rt_target[rt_valid].cpu().numpy()

    sampled_bin = torch.distributions.Categorical(probs=p_decision.clamp_min(1e-8)).sample()
    rt_pred = ((sampled_bin.float() + 0.5) * dt).cpu().numpy()

    bins = np.linspace(0, max_steps * dt, max_steps + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rt_obs, bins=bins, alpha=0.5, density=True, label='Observed', color='tab:blue')
    ax.hist(rt_pred, bins=bins, alpha=0.5, density=True, label='Model (sampled)', color='tab:orange')
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()

    return fig


if __name__ == '__main__':

    path_data = 'weinhardt2026/studies/eckstein2026/data/eckstein2024.csv'
    path_spice = 'weinhardt2026/studies/eckstein2026/params/spice_eckstein2026_rt.pkl'

    max_steps = 40
    t_max = 2.0
    dt = t_max / max_steps

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    prototype = True
    n_participants_prototype = 100
    n_trials_prototype = 100

    dataset_train, dataset_test, info_dataset = get_dataset(path_data, test_blocks=(2,), verbose=True)

    if prototype:
        dataset_train = subset_dataset(dataset_train, n_participants=n_participants_prototype, n_trials=n_trials_prototype)
        if dataset_test is not None:
            dataset_test = subset_dataset(dataset_test, n_participants=n_participants_prototype, n_trials=n_trials_prototype)
        info_dataset['n_participants'] = dataset_train.n_participants

    estimator = SpiceEstimator(
        spice_class=SpiceModel,
        spice_config=CONFIG,
        kwargs_spice_class={'max_steps': max_steps, 't_max': t_max},

        n_actions=info_dataset['n_actions'],
        n_participants=info_dataset['n_participants'],

        loss_fn=make_rt_loss(n_actions=info_dataset['n_actions'], dt=dt, max_steps=max_steps, rt_weight=1.0),
        loss_fn_kwargs={},

        sindy_weight=0.,
        sindy_refit=True,

        epochs=0,
        warmup_steps=0,
        learning_rate=1e-2,
        ensemble_size=8,
        batch_size=256,
        dropout=0.1,

        device=device,
        verbose=True,
        save_path_spice=path_spice,
    )
    
    if estimator.epochs == 0:
        estimator.load_spice(path_spice)
    if estimator.epochs > 0 or estimator.sindy_refit:
        estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    
    print("\n--- Train ---")
    print(evaluate(estimator, dataset_train, info_dataset['n_actions'], dt, max_steps))
    print("\n--- Test ---")
    print(evaluate(estimator, dataset_test, info_dataset['n_actions'], dt, max_steps))

    plot_rt_distribution(estimator, dataset_train, info_dataset['n_actions'], dt, max_steps)
    plot_rt_distribution(estimator, dataset_test, info_dataset['n_actions'], dt, max_steps)
