import sys, os

import torch
from tqdm import tqdm
import numpy as np

from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import convert_dataset, split_data_along_timedim, split_data_along_sessiondim
from spice.resources.bandits import Agent

class RLLSTM(torch.nn.Module):
    
    def __init__(self, n_cells, n_actions):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(n_actions*2, n_cells, batch_first=True, dropout=0.5)
        self.lin_out = torch.nn.Linear(n_cells, n_actions)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.device = torch.device('cpu')
        self.n_actions = n_actions
        self.n_cells = n_cells
        
    def forward(self, inputs, state=None):
        
        x, state = self.lstm(inputs, state)
        logits = self.lin_out(x)
        # logits = self.softmax(logits)
        return logits, state
    
    def to(self, device):
        super().to(device)
        self.device = device
        
        return self
    

class AgentLSTM(Agent):
    """A class that allows running a pretrained LSTM as an agent.

    Attributes:
        model: A PyTorch module representing the LSTM architecture
    """

    def __init__(
        self,
        model_rnn: RLLSTM,
        n_actions: int = 2,
        device = torch.device('cpu'),
        deterministic: bool = True,
        ):
        """Initialize the agent network.

        Args:
            model: A PyTorch module representing the RNN architecture
            n_actions: number of permitted actions (default = 2)
        """

        super().__init__(n_actions=n_actions, deterministic=deterministic)
        
        assert isinstance(model_rnn, RLLSTM), "The passed model is not an instance of RLLSTM."
        
        self._model = model_rnn
        self._model = self._model.to(device)
        self._model.eval()

        self._state = {'x_value_reward': np.zeros((n_actions))}

    def new_sess(self, *args, **kwargs):
        """Reset the network for the beginning of a new session."""    
        self._state = {'x_value_reward': np.zeros((self._n_actions))}
        self._hidden_state = torch.zeros((1, self._model.n_cells)).to(self._model.device)
        self._cell_state = torch.zeros((1, self._model.n_cells)).to(self._model.device)

    def update(self, choice: float, reward: float, **kwargs):
        choice = torch.eye(self._n_actions)[int(choice)]
        xs = torch.concat([choice, torch.tensor(reward)]).view(1, -1).to(device=self._model.device)
        with torch.no_grad():
            logits, state = self._model(xs, self.get_state())
        self.set_state(logits, *state)

    def set_state(self, logits: np.ndarray, hidden_state: torch.Tensor, cell_state: torch.Tensor):
        self._state['x_value_reward'] = logits.detach().cpu().numpy().reshape(-1)
        self._hidden_state = hidden_state.to(self._model.device).detach()
        self._cell_state = cell_state.to(self._model.device).detach()

    def get_state(self):
        return self._hidden_state, self._cell_state

    @property
    def q(self):
        return self._state['x_value_reward']

  
def setup_agent_lstm(path_model: str) -> AgentLSTM:
    state_dict = torch.load(path_model, map_location=torch.device('cpu'))
    
    n_cells = state_dict['lin_out.weight'].shape[1]
    n_actions = state_dict['lin_out.weight'].shape[0]
    
    lstm = RLLSTM(n_cells=n_cells, n_actions=n_actions)
    lstm.load_state_dict(state_dict=state_dict)
    agent = AgentLSTM(model_rnn=lstm, n_actions=n_actions)
    return agent

def training(dataset_training: SpiceDataset, lstm: RLLSTM, optimizer: torch.optim.Optimizer, epochs = 3000, dataset_test: SpiceDataset = None, criterion = torch.nn.CrossEntropyLoss()):
    
    for e in range(epochs):
        
        mask = (dataset_training.xs[..., :1] > -1).to(lstm.device)
        # prediction
        ys_pred, state = lstm(dataset_training.xs[..., :lstm.n_actions*2].to(lstm.device))
        
        # loss computation
        loss = criterion(
            (ys_pred * mask).reshape(-1, lstm.n_actions),
            torch.argmax((
                dataset_training.ys.to(lstm.device)  * mask
                ).reshape(-1, lstm.n_actions), dim=1),
            )
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if dataset_test is not None:
            with torch.no_grad():
                mask = (dataset_test.xs[..., :1] > -1).to(lstm.device)
                # prediction
                
                if dataset_training.xs.shape[0] == dataset_test.xs.shape[0]:
                    input_lstm = torch.concat((dataset_training.xs[..., :lstm.n_actions*2], dataset_test.xs[..., :lstm.n_actions*2]), dim=1).to(lstm.device)
                else:
                    input_lstm = dataset_test.xs[..., :lstm.n_actions*2].to(lstm.device)
                ys_pred, state = lstm(input_lstm)
                
                # loss computation
                loss_test = criterion(
                    (ys_pred[:, -dataset_test.xs.shape[1]:] * mask).reshape(-1, lstm.n_actions),
                    torch.argmax((
                        dataset_test.ys.to(lstm.device)  * mask
                        ).reshape(-1, lstm.n_actions), dim=1),
                    ).item()
        
        print(f"{e+1}/{epochs}: L(Train) = {loss.item():.5f}; L(Test) = {loss_test:.5f}")
    
    return lstm

def main(path_save_model:str, path_data: str, n_actions: int, n_cells: int, n_epochs: int, lr: float, split_ratio: float, device=torch.device('cpu')):
    
    if isinstance(split_ratio, float):
        dataset_training, dataset_test = split_data_along_timedim(convert_dataset(path_data)[0], split_ratio=split_ratio)
    else:
        dataset_training, dataset_test = split_data_along_sessiondim(convert_dataset(path_data)[0], list_test_sessions=split_ratio)
        
    lstm = RLLSTM(n_cells=n_cells, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
    
    print('Training LSTM...')
    lstm = training(dataset_training=dataset_training, dataset_test=dataset_test, lstm=lstm, optimizer=optimizer, epochs=n_epochs)
    
    torch.save(lstm.state_dict(), path_save_model)
    
    print('Training LSTM done!')
    
if __name__=='__main__':
    
    # dataset_name = 'eckstein2022'
    # split_ratio = 0.8
    
    # dataset_name = 'dezfouli2019'
    # split_ratio = [3, 6, 9]
    
    dataset_name = 'gershmanB2018'
    split_ratio = [4, 8, 12, 16]
    
    path_model_save = f'params/{dataset_name}/lstm_{dataset_name}.pkl'
    path_data = f'data/{dataset_name}/{dataset_name}.csv'
    n_actions = 2
    n_cells = 16
    n_epochs = 2000
    lr = 1e-3
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    main(path_save_model=path_model_save, path_data=path_data, n_actions=n_actions, n_cells=n_cells, n_epochs=n_epochs, lr=lr, split_ratio=split_ratio, device=device)