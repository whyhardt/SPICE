import torch

from .rnn import BaseRNN
from .spice_utils import SpiceConfig
from ..utils.convert_dataset import csv_to_dataset


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
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=self.hidden_size, batch_first=True)
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
        self.max_steps = self.max_steps
        self.t_max = t_max
        self.dt = self.t_max / self.max_steps
        
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=self.embedding_size, dropout=dropout)
        
        self.setup_module(key_module='drift', input_size=1+self.embedding_size, dropout=self.dropout)
        self.setup_module(key_module='diffusion', input_size=0+self.embedding_size, dropout=self.dropout)
        
        self.traces = {
            'evidence': torch.zeros(1, self.max_steps, self.n_actions),
            'drift': torch.zeros(1, self.max_steps, self.n_actions),
            'diffusion': torch.zeros(1, self.max_steps, self.n_actions),
        }
        
    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None, batch_first: bool = False):
        
        # inputs carries only the stimulus signal and IDs
        stimulus = inputs[..., 0].unsqueeze(-1).repeat(1, 1, self.n_actions)
        
        # experiment_ids = inputs[:, 0, -2]
        participants_ids = inputs[:, 0, -1]
        participant_embeddings = self.participant_embedding(participants_ids)
        
        self.traces['drift'] = self.call_module(
            key_module='drift',
            key_state='drift',
            action_mask=None,
            inputs=(stimulus),
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
        
    def init_state(self, batch_size=1):
        super().init_state(batch_size)
        self.traces = {key: torch.zeros((batch_size, self.max_steps, self.n_actions), device=self.device, requires_grad=False) for key in self.traces}
        return self.get_state()
    
    def get_state(self, detach=False):
        return self.traces
    


if __name__=='__main__':
    
    dataset = csv_to_dataset(
        file='ddm/ddm.csv',
        additional_inputs=('stimulus'),
    )
    model = DDMRNN()