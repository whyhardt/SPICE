import torch

from spice import SpiceEstimator, BaseRNN, SpiceConfig, SpiceDataset, csv_to_dataset


SPICE_CONFIG = SpiceConfig(
            library_setup={
                'drift': ['stimulus'],
                'diffusion': [],
            },
            memory_state={
                'evidence': 0.5,
                'drift': 0,
                'diffusion': 0,
            }
        )


def wasserstein_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein distance between sorted signed RT distributions.

    Sorts both predicted and target signed RTs and computes the mean absolute
    difference between corresponding quantiles.
    """
    
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    return torch.mean(torch.abs(prediction.sort()[0] - target.sort()[0]))


class ThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory, x, b, dt, tnd=0.2):
        mask = trajectory >= 0
        decision_time = mask.float().argmax(dim=1).int()
        decision_time[mask.sum(dim=1) == 0] = torch.tensor(trajectory.shape[1] - 1, dtype=torch.int)

        evidences = torch.gather(x, dim=1, index=decision_time.unsqueeze(-1).to(dtype=torch.int64)).squeeze(-1)
        times = decision_time * dt + tnd
        signs = torch.sign(evidences)
        times = times * signs

        ctx.save_for_backward(dsdt_trajectory, decision_time, trajectory, signs)
        return times, evidences, decision_time

    @staticmethod
    def backward(ctx, grad_rt, grad_evidences, grad_decision_time):
        dsdt_trajectory, decision_times, trajectory, signs = ctx.saved_tensors
        decision_times = decision_times.squeeze()
        mask = trajectory >= 0
        idx1 = (mask.sum(dim=1) == 0).long()
        idx2 = dsdt_trajectory[torch.arange(dsdt_trajectory.size(0)), decision_times.long()] < 0
        idx = torch.logical_and(idx1.bool(), idx2.bool()).squeeze()
        grads = torch.zeros_like(dsdt_trajectory)
        batch_indices = torch.arange(decision_times.size(0)).to(decision_times.device)

        grads[batch_indices, decision_times.long()] = -1.0 / (dsdt_trajectory[
                                                                  batch_indices, decision_times.long()] + 1e-6)
        grads[batch_indices[idx], decision_times[idx].long()] = 1e-6 / (
                trajectory[batch_indices[idx], decision_times[idx].long()] + 1e-6
        )
        grads = grads * grad_rt.unsqueeze(1).expand_as(grads) * signs.unsqueeze(-1)
        return grads, None, None, None, None, None, None


class DDMRNN(BaseRNN):

    def __init__(
        self,
        n_participants: int,
        n_experiments: int,
        max_steps: int = 100,
        t_max: float = 5.0,
        device: torch.device = torch.device('cpu'),
        **kwargs,
        ):

        super().__init__(
            n_actions=2,
            spice_config=SPICE_CONFIG,
            n_participants=n_participants,
            n_experiments=n_experiments,
            n_reward_features=0,
            sindy_polynomial_degree=2,
            sindy_ensemble_size=1,
            device=device,
            )

        self.dropout = 0.1
        self.max_steps = max_steps
        self.t_max = t_max
        self.dt = t_max / max_steps
        self.threshold = 1.0
        self.init_time = 0.2
        self.init_traces()

        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout)

        self.setup_module(key_module='drift', input_size=1+self.embedding_size, dropout=self.dropout)
        # self.setup_module(key_module='diffusion', input_size=0+self.embedding_size, dropout=self.dropout)

    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None, batch_first: bool = False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        experiment_ids = spice_signals.experiment_ids
        participant_ids = spice_signals.participant_ids
        participant_emb = self.participant_embedding(participant_ids)

        # Single outer timestep (T_out=1): additional_inputs[0] is [W=max_steps, B, n_add]
        # additional_inputs layout: abs_time(0), stimulus_strength(1), choice(2), response_time(3)
        stimulus = spice_signals.additional_inputs[0, :, :, 1].unsqueeze(-1).expand(-1, -1, self.n_actions)
        # [W, B, n_actions]

        self.traces['drift'] = self.call_module(
            key_module='drift',
            key_state='drift',
            action_mask=None,
            inputs=(stimulus,),
            participant_index=participant_ids,
            participant_embedding=participant_emb,
            # activation_rnn=torch.exp,
        )   # [W, B, n_actions]
        
        # self.traces['diffusion'] = self.call_module(
        #     key_module='diffusion',
        #     key_state='diffusion',
        #     action_mask=None,
        #     inputs=None,
        #     participant_index=participant_ids,
        #     participant_embedding=participant_emb,
        #     activation_rnn=torch.exp,
        # )   # [1, B, n_actions] — constant per trial, broadcasts across W
        self.traces['diffusion'] = torch.ones_like(self.traces['drift'])
        
        batch_size = participant_ids.shape[0]
        epsilon = torch.randn((self.max_steps, batch_size, 1), device=self.device, requires_grad=False)

        # Single-accumulator DDM: use first action's drift/diffusion
        drift = self.traces['drift'][:, :, 0:1]          # [W, B, 1]
        diffusion = self.traces['diffusion'][:, :, 0:1]  # [1, B, 1]

        evidence_t = drift * self.dt + diffusion * epsilon * torch.sqrt(torch.tensor(self.dt))
        self.traces['evidence'] = torch.cumsum(evidence_t, dim=0)  # [W, B, 1]

        # ThresholdFunction expects batch-first: [B, max_steps, 1]
        evidence_bf = self.traces['evidence'].permute(1, 0, 2)
        evidence_t_bf = evidence_t.permute(1, 0, 2)

        response_time, final_evidence, decision_times = ThresholdFunction.apply(
            torch.abs(evidence_bf) - self.threshold,
            evidence_t_bf,
            evidence_bf,
            self.threshold,
            self.dt,
            self.init_time,
            )
        
        output = torch.concat((response_time, experiment_ids.view(-1, 1), participant_ids.view(-1, 1)), dim=0)
        
        # Return [B, T_out=1] to match training loop expectations
        return output, self.get_state()

    def init_traces(self, batch_size=1):

        self.traces = {
            'evidence': torch.zeros(self.max_steps, batch_size, self.n_actions),
            'drift': torch.zeros(self.max_steps, batch_size, self.n_actions),
            'diffusion': torch.zeros(1, batch_size, self.n_actions),
        }

        return self.traces


# --------------------------------------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# --------------------------------------------------------------------------------------------------------------
import numpy as np

def simulate_ddm(
    n_participants=1,
    n_trials=1000,
    t_max=5.0,
    max_steps=100,
    drift_rate=1.0,
    diffusion_rate=1.0,
    threshold=1.0,
    non_decision_time=0.2,
    save_path=None,
    device=None,
):
    """Simulate DDM evidence accumulation and return data as a SpiceDataset.

    xs shape: (n_trials, 1, max_steps, 9) — 4D with outer_ts=1, within_ts=max_steps.
    xs features: (choice_0, choice_1, abs_time, stimulus_strength, choice, response_time, trial, experiment_id, participant_id)
    ys shape: (n_trials, 1, 1, 1) — signed response time (one scalar per trial).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt = t_max / max_steps
    n_total = n_participants * n_trials
    time_arr = torch.arange(max_steps, device=device) * dt

    # features: choice_0(0), choice_1(1), abs_time(2), stimulus_strength(3),
    #           choice(4), response_time(5), trial(6), experiment_id(7), participant_id(8)
    xs = torch.full((n_total, max_steps, 10), float('nan'), device=device)

    # sample stimulus direction per trial: (n_total,)
    stimuli = torch.full((n_total,), drift_rate, device=device)

    # simulate evidence accumulation: (n_total, max_steps)
    noise = torch.randn(n_total, max_steps, device=device)
    increments = stimuli.unsqueeze(1) * dt + diffusion_rate * (dt ** 0.5) * noise
    evidence = torch.cumsum(increments, dim=1)

    # find first boundary crossing per trial
    upper_mask = evidence >= threshold
    lower_mask = evidence <= -threshold
    has_upper = upper_mask.any(dim=1)
    has_lower = lower_mask.any(dim=1)
    # argmax on bool returns first True; returns 0 if none True
    first_upper = torch.where(has_upper, upper_mask.to(torch.int).argmax(dim=1), max_steps)
    first_lower = torch.where(has_lower, lower_mask.to(torch.int).argmax(dim=1), max_steps)

    # choice=1 if upper boundary crossed first (or equal)
    choice = (first_upper <= first_lower).to(torch.long)
    decision_step = torch.minimum(first_upper, first_lower)

    # handle trials that never crossed a boundary
    no_crossing = decision_step >= max_steps
    choice[no_crossing] = (evidence[no_crossing, -1] > 0).to(torch.long)

    response_time = torch.where(no_crossing, t_max + non_decision_time, decision_step * dt + non_decision_time)
    signed_rt = torch.where(choice == 1, response_time, -response_time)

    # trial and participant indices
    participant_ids = torch.arange(n_participants, device=device).repeat_interleave(n_trials)
    trial_ids = torch.arange(n_trials, device=device).repeat(n_participants)

    # fill constant features across all timesteps
    trial_idx = torch.arange(n_total, device=device)
    xs[trial_idx, :, choice] = 1.0                          # chosen action one-hot
    xs[trial_idx, :, 1 - choice] = 0.0                      # other action one-hot
    xs[:, :, 2] = time_arr.unsqueeze(0)                     # abs_time
    xs[:, :, 3] = stimuli.unsqueeze(1)                      # stimulus_strength
    xs[:, :, 4] = choice.unsqueeze(1).float()               # choice index
    xs[:, :, 5] = signed_rt.unsqueeze(1)                    # signed response time
    xs[:, :, 6] = trial_ids.unsqueeze(1).float()            # trial
    xs[:, :, 7] = 0                                         # block
    xs[:, :, 8] = 0                                         # experiment_id
    xs[:, :, 9] = participant_ids.unsqueeze(1).float()      # participant_id
    
    if save_path is not None:
        import pandas as pd
        columns = ['choice_0', 'choice_1', 'abs_time', 'stimulus_strength',
                    'choice', 'response_time', 'trial', 'experiment_id', 'participant_id']
        rows = xs.reshape(-1, 9).cpu().numpy()
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(save_path, index=False)

    # Reshape to 4D: (n_trials, outer_ts=1, within_ts=max_steps, features)
    xs = xs.unsqueeze(1).cpu()                               # [n_total, 1, max_steps, 9]
    ys = signed_rt.reshape(n_total, 1, 1, 1).cpu()           # [n_total, 1, 1, 1]
    
    return SpiceDataset(xs=xs, ys=ys)


if __name__=='__main__':

    max_steps = 100
    t_max = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    drift_rates = [0., 0.5, 1.0]
    
    datasets = []
    xs, ys = [], []
    for index_rate, rate in enumerate(drift_rates):
        dataset = simulate_ddm(
            n_participants=1,
            n_trials=1000,
            t_max=t_max,
            max_steps=max_steps,
            drift_rate=rate,
            diffusion_rate=1.0,
            threshold=1.0,
            non_decision_time=0.2,
        )
        
        xs_rate = dataset.xs
        xs_rate[..., -1] = index_rate
        
        xs.append(xs_rate)
        ys.append(dataset.ys)
    dataset = SpiceDataset(torch.concat(xs), torch.concat(ys))
    
    spice_estimator = SpiceEstimator(
        spice_config=SPICE_CONFIG,
        rnn_class=DDMRNN,
        kwargs_rnn_class={
            'max_steps': max_steps,
            't_max': t_max,
        },
        loss_fn=wasserstein_loss,
        
        n_actions=2,
        n_participants=len(drift_rates),

        epochs=1000,
        learning_rate=0.0001,
        bagging=False,
        device=device,

        sindy_weight=0.,
        
        verbose=True,
    )

    spice_estimator.fit(dataset.xs, dataset.ys)
    
    model = spice_estimator.rnn_agent.model
    signed_rt, _ = model(dataset.xs.to(device), batch_first=True)
    
    import matplotlib.pyplot as plt
    
    plt.hist(dataset.xs[:1000, 0, 0, 5].cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Real", color='tab:blue')
    plt.hist(signed_rt[:1000, 0].detach().cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Fake", color='tab:orange')
    plt.legend()
    plt.show()
    
    plt.hist(dataset.xs[1000:2000, 0, 0, 5].cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Real", color='tab:blue')
    plt.hist(signed_rt[1000:2000, 0].detach().cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Fake", color='tab:orange')
    plt.legend()
    plt.show()
    
    plt.hist(dataset.xs[2000:3000, 0, 0, 5].cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Real", color='tab:blue')
    plt.hist(signed_rt[2000:3000, 0].detach().cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Fake", color='tab:orange')
    plt.legend()
    plt.show()
    