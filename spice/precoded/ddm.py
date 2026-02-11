import torch

from spice import SpiceEstimator, BaseRNN, SpiceConfig, SpiceDataset, csv_to_dataset


SPICE_CONFIG = SpiceConfig(
            library_setup={
                'drift': ['stimulus'],
                # 'diffusion': [],
            },
            memory_state={
                # 'evidence': 0.,
                'drift': 0,
                # 'diffusion': 1,
            }
        )


def wasserstein_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein distance between sorted signed RT distributions.

    Sorts both predicted and target signed RTs and computes the mean absolute
    difference between corresponding quantiles.
    """
    
    # Extract components
    pred_values = prediction[:, 0]  # shape: (batch,)
    experiment_ids = prediction[:, 1]  # shape: (batch,)
    participant_ids = prediction[:, 2]  # shape: (batch,)
    target_values = target[:, 0]  # shape: (batch,)
    
    # Create unique group identifier
    # Combine participant and experiment IDs into a single group key
    group_key = participant_ids * 1000 + experiment_ids  # adjust multiplier if needed
    
    # Get unique groups and inverse indices
    unique_groups, inverse_indices = torch.unique(group_key, return_inverse=True)

    # Compute Wasserstein loss for each group
    wasserstein_losses = []
    for group_idx in range(len(unique_groups)):
        # Get mask for current group
        mask = inverse_indices == group_idx
        
        # Extract values for this group
        group_pred = pred_values[mask]
        group_target = target_values[mask]
        
        # Sort and compute Wasserstein distance
        sorted_pred = group_pred.sort()[0]
        sorted_target = group_target.sort()[0]
        
        wasserstein_loss = torch.mean(torch.abs(sorted_pred - sorted_target))
        wasserstein_losses.append(wasserstein_loss)

    # Average across all groups
    total_loss = torch.mean(torch.stack(wasserstein_losses))
    
    return total_loss


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

        self.dropout = 0.
        self.max_steps = max_steps
        self.t_max = t_max
        self.dt = t_max / max_steps
        self.threshold = 1.0
        self.init_time = 0.2
        
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout)

        self.setup_module(key_module='drift', input_size=1+self.embedding_size, dropout=self.dropout)

    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None, batch_first: bool = False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        experiment_ids = spice_signals.experiment_ids
        participant_ids = spice_signals.participant_ids
        participant_emb = self.participant_embedding(participant_ids)

        # Single outer timestep (T_out=1): additional_inputs[0] is [W=max_steps, B, n_add]
        # additional_inputs layout: stimulus_strength(0)
        stimulus = spice_signals.additional_inputs[0, :, :, 0].unsqueeze(-1).expand(-1, -1, self.n_actions)
        # [W, B, n_actions]

        self.call_module(
            key_module='drift',
            key_state='drift',
            action_mask=None,
            inputs=(stimulus,),
            participant_index=participant_ids,
            participant_embedding=participant_emb,
        )   # [W, B, n_actions]
        
        batch_size = participant_ids.shape[0]
        epsilon = torch.randn((self.max_steps, batch_size, 1), device=self.device, requires_grad=False)

        # Single-accumulator DDM: use first action's drift/diffusion
        drift = self.state['drift'][:, :, 0:1]          # [W, B, 1]
        # diffusion = self.state['diffusion'][:, :, 0:1]  # [W, B, 1]
        diffusion = torch.ones_like(drift)
        
        evidence_t = drift * self.dt + diffusion * epsilon * torch.sqrt(torch.tensor(self.dt))
        evidence = torch.cumsum(evidence_t, dim=0)  # [W, B, 1]

        # ThresholdFunction expects batch-first: [B, max_steps, 1]
        evidence_bf = evidence.permute(1, 0, 2)
        evidence_t_bf = evidence_t.permute(1, 0, 2)

        response_time, final_evidence, decision_times = ThresholdFunction.apply(
            torch.abs(evidence_bf) - self.threshold,
            evidence_t_bf,
            evidence_bf,
            self.threshold,
            self.dt,
            self.init_time,
            )
        
        output = torch.concat((
            response_time, 
            experiment_ids.view(-1, 1), 
            participant_ids.view(-1, 1)
            ), dim=-1).reshape(-1, 1, 1, 3).repeat(1, 1, self.max_steps, 1)  # add trial dimension
        
        # Return [B, T_in=1, T_out=max_steps, 3] to match training loop expectations
        return output, self.get_state()


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

    # features: choice_0(0), choice_1(1), stimulus_strength(2), within trial time(3), trial(4), block(5), experiment_id(6), participant_id(7)
    xs = torch.full((n_total, max_steps, 8), float('nan'), device=device)

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
    xs[:, :, 2] = stimuli.unsqueeze(1)                      # stimulus_strength
    xs[:, :, 3] = time_arr.unsqueeze(0)                     # within trial time
    xs[:, :, 4] = trial_ids.unsqueeze(1).float()            # trial
    xs[:, :, 5] = 0                                         # block
    xs[:, :, 6] = 0                                         # experiment_id
    xs[:, :, 7] = participant_ids.unsqueeze(1).float()      # participant_id
    
    if save_path is not None:
        import pandas as pd
        columns = ['choice_0', 'choice_1', 'abs_time', 'stimulus_strength',
                    'choice', 'response_time', 'trial', 'experiment_id', 'participant_id']
        rows = xs.reshape(-1, 8).cpu().numpy()
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(save_path, index=False)

    # Reshape to 4D: (n_trials, outer_ts=1, within_ts=max_steps, features)
    xs = xs.unsqueeze(1).cpu()                               # [n_total, 1, max_steps, 10]
    ys = signed_rt.reshape(n_total, 1, 1, 1).repeat(1, 1, max_steps, 1).cpu()           # [n_total, 1, 1, 1]
    
    return SpiceDataset(xs=xs, ys=ys)


if __name__=='__main__':

    # --------------------------------------------------------------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------------------------------------------------------------
    
    max_steps = 100
    t_max = 5
    n_trials = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    drift_rates = [0.5, 1.0, 0.2, 1.0, 0.5, 0.2]#[0.2, 0.5, 1.0]
    n_participants = len(drift_rates)
    
    datasets = []
    xs, ys = [], []
    for index_rate, rate in enumerate(drift_rates):
        dataset = simulate_ddm(
            n_participants=1,
            n_trials=n_trials,
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
            'max_steps': dataset.xs.shape[2],
            't_max': t_max,
        },
        loss_fn=wasserstein_loss,
        
        n_actions=2,
        n_participants=n_participants,

        epochs=2000,
        sindy_epochs=2000,
        warmup_steps=1000,
        learning_rate=0.00001,
        bagging=False,
        scheduler=False,
        device=device,

        sindy_weight=0.00001,
        sindy_l2_lambda=0.0001,
        sindy_library_polynomial_degree=2,
        
        # save_path_spice='spice_ddm.pkl',
        verbose=True,
    )
    
    spice_estimator.fit(data=dataset.xs, targets=dataset.ys, data_test=dataset.xs, target_test=dataset.ys)
    
    # --------------------------------------------------------------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------------------------------------------------------------
    
    model_rnn = spice_estimator.rnn_agent.model
    model_spice = spice_estimator.spice_agent.model
    signed_rt_rnn, state_rnn = model_rnn(dataset.xs.to(device), batch_first=True)
    if spice_estimator.sindy_weight > 0:
        signed_rt_spice, state_spice = model_spice(dataset.xs.to(device), batch_first=True)
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(n_participants, 2)
    
    for index in range(n_participants):
        axs[index, 1].hist(dataset.ys[index*n_trials:(index+1)*n_trials, 0, 0, 0].cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="Real", color='tab:blue')
        axs[index, 1].hist(signed_rt_rnn[index*n_trials:(index+1)*n_trials, 0, 0, 0].detach().cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="RNN", color='tab:orange')
        
        axs[index, 0].plot(dataset.xs[500+n_trials*index, 0, :, 2].detach().cpu().numpy(), label="Real", color='tab:blue')
        axs[index, 0].plot(state_rnn['drift'][:, 500+n_trials*index, 0].detach().cpu().numpy(), label="RNN", color='tab:orange')
        
        if spice_estimator.sindy_weight > 0:
            model_spice.print_spice_model(participant_id=index)
            axs[index, 1].hist(signed_rt_spice[index*n_trials:(index+1)*n_trials, 0, 0, 0].detach().cpu().numpy(), bins=50, range=(-5.2, 5.2), alpha=0.4, label="SPICE", color='tab:green')
            axs[index, 0].plot(state_spice['drift'][:, 500+n_trials*index, 0].detach().cpu().numpy(), label="SPICE", color='tab:green')
        
    axs[0, 1].legend()
    plt.show()