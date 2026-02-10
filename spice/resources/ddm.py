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


class GRUModule(torch.nn.Module):
    
    def __init__(self, input_size: int, dropout: float = 0., **kwargs):
        super().__init__()
        
        self.hidden_size = input_size+8
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=self.hidden_size, batch_first=False)
        self.linear_out = torch.nn.Linear(in_features=self.hidden_size, out_features=1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor = None):
        
        if state is None:
            state = torch.zeros((inputs.shape[1], self.hidden_size))
        
        y, state = self.gru(inputs, state)
        y = self.dropout(y)
        y = self.linear_out(y)
        return y, state


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
            sindy_polynomial_degree=2,
            sindy_ensemble_size=1,
            device=device,
            )
        
        self.dropout = 0.1
        self.max_steps = max_steps
        self.t_max = t_max
        self.dt = t_max / max_steps
        self.init_traces()
        
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout)
        
        self.setup_module(key_module='drift', input_size=1+self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='diffusion', input_size=0+self.embedding_size, dropout=self.dropout)
        
    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None, batch_first: bool = False):
        
        if batch_first:
            inputs = inputs.swapaxes(1, 0)
        
        # inputs carries only the stimulus signal and IDs
        stimulus = inputs[..., 0].unsqueeze(-1).repeat(1, 1, self.n_actions)
        
        # experiment_ids = inputs[0, :, -2]
        participants_ids = inputs[0, :, -1].long()
        participant_embeddings = self.participant_embedding(participants_ids)
        
        self.traces['drift'] = self.call_module(
            key_module='drift',
            key_state='drift',
            action_mask=None,
            inputs=(stimulus,),
            participant_index=participants_ids,
            participant_embedding=participant_embeddings,
            activation_rnn=torch.exp,
        )
        
        self.traces['diffusion'] = self.call_module(
            key_module='diffusion',
            key_state='diffusion',
            action_mask=None,
            inputs=None,
            participant_index=participants_ids,
            participant_embedding=participant_embeddings,
            activation_rnn=torch.exp,
        )
        
        epsilon = torch.randn((inputs.shape[1], self.max_steps, 1), device=self.device, requires_grad=False)
        
        evidence_t = self.traces['drift'] * self.dt + self.traces['diffusion'] * epsilon * torch.sqrt(self.dt)
        self.traces['evidence'] = torch.cumsum(evidence_t, dim=1)
        
        response_time, final_evidence, decision_times = ThresholdFunction.apply(
            trajectory=torch.abs(self.traces['evidence']) - self.threshold,
            dsdt_trajectory=evidence_t,
            x=self.traces['evidence'],
            b=self.threshold,
            dt=self.dt,
            tnd=self.init_time,
            )
        
        signed_response_time = response_time * torch.sign(final_evidence)
        
        return signed_response_time, self.get_state()
    
    def init_traces(self, batch_size=1):
        
        self.traces = {
            'evidence': torch.zeros(batch_size, self.max_steps, self.n_actions),
            'drift': torch.zeros(batch_size, self.max_steps, self.n_actions),
            'diffusion': torch.zeros(batch_size, self.max_steps, self.n_actions),
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

    xs features: (choice_0, choice_1, abs_time, stimulus_strength, choice, response_time, trial, experiment_id, participant_id)
    All features are constant across timesteps within a trial.
    ys: signed response time.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt = t_max / max_steps
    n_total = n_participants * n_trials
    time_arr = torch.arange(max_steps, device=device) * dt

    # features: choice_0(0), choice_1(1), abs_time(2), stimulus_strength(3),
    #           choice(4), response_time(5), trial(6), experiment_id(7), participant_id(8)
    xs = torch.full((n_total, max_steps, 9), float('nan'), device=device)

    # sample stimulus direction per trial: (n_total,)
    stimuli = torch.full((n_total,), 0.5, device=device)

    # simulate evidence accumulation: (n_total, max_steps)
    noise = torch.randn(n_total, max_steps, device=device)
    increments = drift_rate * stimuli.unsqueeze(1) * dt + diffusion_rate * (dt ** 0.5) * noise
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
    xs[:, :, 7] = 0                                         # experiment_id
    xs[:, :, 8] = participant_ids.unsqueeze(1).float()      # participant_id
    ys = xs[:, :, 5:6].clone()

    if save_path is not None:
        import pandas as pd
        columns = ['choice_0', 'choice_1', 'abs_time', 'stimulus_strength',
                    'choice', 'response_time', 'trial', 'experiment_id', 'participant_id']
        rows = xs.reshape(-1, 9).cpu().numpy()
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(save_path, index=False)

    # move to cpu for SpiceDataset
    return SpiceDataset(xs=xs.cpu(), ys=ys.cpu())
    

if __name__=='__main__':
    
    # dataset = csv_to_dataset(
    #     file='ddm/ddm.csv',
    #     additional_inputs=('stimulus'),
    # )
    
    max_steps = 100
    t_max = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = simulate_ddm(
        n_participants=1,
        n_trials=1000,
        t_max=t_max,
        max_steps=max_steps,
        drift_rate=1.0,
        diffusion_rate=1.0,
        threshold=1.0,
        non_decision_time=0.2,
    )
    
    # Check dataset
    # print("Choices:")
    # print(dataset.xs[:, 0, 4])
    # print("RTs")
    # print(dataset.ys[:, 0].sort(dim=0)[0].reshape(-1))
    
    # import matplotlib.pyplot as plt
    # plt.hist(dataset.ys[:, 0, 0], bins=50, range=(-5.2, 5.2))
    # plt.show()
    
    spice_estimator = SpiceEstimator(
        spice_config=SPICE_CONFIG,
        rnn_class=DDMRNN,
        kwargs_rnn_class={
            'max_steps': max_steps,
            't_max': t_max,  
        },
        
        n_actions=2,
        n_participants=1,
        
        epochs=1,
        device=device,
        
        sindy_weight=0.,
    )
    
    spice_estimator.fit(dataset.xs, dataset.ys)