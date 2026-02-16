import torch

from spice import SpiceDataset, csv_to_dataset, split_data_along_sessiondim, split_data_along_timedim, Agent, cross_entropy_loss


class GRU(torch.nn.Module):
    
    def __init__(self, n_actions, additional_inputs: int = 0, hidden_size: int = 16, **kwargs):
        super().__init__()
        
        self.gru_features = hidden_size
        self.n_actions = n_actions
        self.additional_inputs = additional_inputs
        
        self.linear_in = torch.nn.Linear(in_features=n_actions+1+additional_inputs, out_features=hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear_out = torch.nn.Linear(in_features=hidden_size, out_features=n_actions)
        
    def forward(self, inputs, state=None):
        
        actions = inputs[..., :self.n_actions]
        rewards = inputs[..., self.n_actions:2*self.n_actions].nan_to_num(0).sum(dim=-1, keepdims=True)
        additional_inputs = inputs[..., self.n_actions*2:self.n_actions*2+self.additional_inputs]
        inputs = torch.concat((actions, rewards, additional_inputs), dim=-1)
        
        if state is not None and len(inputs.shape) == 3:
            state = state.reshape(1, 1, self.gru_features)
        
        y = self.linear_in(inputs[:, 0].nan_to_num(0))
        y = self.dropout(y)
        y, state = self.gru(y, state)
        y = self.dropout(y)
        y = self.linear_out(y)
        return y, state

  
def setup_agent_gru(path_model: str, gru: torch.nn.Module = None) -> Agent:
    state_dict = torch.load(path_model, map_location=torch.device('cpu'))
    
    hidden_size = state_dict['linear_in.weight'].shape[0]
    n_actions = state_dict['linear_out.weight'].shape[0]
    additional_inputs = state_dict['linear_in.weight'].shape[1] - 1 - n_actions
    
    if gru is None:
        gru = GRU(n_actions=n_actions, hidden_size=hidden_size, additional_inputs=additional_inputs)
    gru.load_state_dict(state_dict=state_dict)
    agent = Agent(model_rnn=gru, n_actions=gru.n_actions)
    return agent

def training(
    gru: GRU, 
    optimizer: torch.optim.Optimizer, 
    dataset_train: SpiceDataset, 
    dataset_test: SpiceDataset = None, 
    epochs = 3000,
    batch_size = None,
    criterion = cross_entropy_loss,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
    
    n_actions = dataset_train.ys.shape[-1]
    batch_size = min(dataset_train.xs.shape[0], batch_size) if batch_size is not None else dataset_train.xs.shape[0]
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        gru.train()
        optimizer.zero_grad()
        
        # random_indexes = torch.randint(dataset_train.xs.shape[0], (1, dataset_train.xs.shape[0]))[0]
        
        # xs_train = dataset_train.xs[random_indexes].to(device)
        # ys_train = dataset_train.ys[random_indexes].to(device)
        # xs_test = dataset_test.xs.to(device)
        # ys_test = dataset_test.ys.to(device)
        
        for batch in dataloader:
            
            xs, ys = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            logits, _ = gru(xs)
            
            # Reshape for loss computation
            nan_mask = ~torch.isnan(xs[..., :gru.n_actions].sum(dim=-1))
            logits = logits[nan_mask]
            labels = ys[nan_mask]
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        msg = f"Epoch {epoch+1}/{epochs}: L(Train): {loss.item()}"
        
        # test data
        if dataset_test is not None:
            gru.eval()
            with torch.no_grad():
                logits_test, _ = gru(dataset_test.xs.to(device))
                nan_mask = ~torch.isnan(dataset_test.xs[..., :gru.n_actions].sum(dim=-1))
                logits_test = logits_test[nan_mask].to(device)
                # labels_test = torch.argmax(dataset_test.ys[..., :n_actions].to(device).reshape(-1, n_actions)[nan_mask], dim=-1).reshape(-1).long()
                labels_test = dataset_test.ys[nan_mask].to(device)
                loss_test = criterion(logits_test, labels_test)
            gru.train()
            
            msg += f"; L(Test): {loss_test.item()}"
        
        print(msg)
        
    return gru

def main(path_save_model:str, path_data: str, epochs: int, lr: float, split_ratio: float, device=torch.device('cpu')):
    
    if isinstance(split_ratio, float):
        dataset_training, dataset_test = split_data_along_timedim(csv_to_dataset(path_data), split_ratio=split_ratio)
    else:
        dataset_training, dataset_test = split_data_along_sessiondim(csv_to_dataset(path_data), list_test_sessions=split_ratio)
    
    n_actions = dataset_training.ys.shape[-1]
    n_participants = len(dataset_training.xs[..., -1].unique())
    
    gru = GRU(input_size=n_actions+1, n_actions=n_actions, n_participants=n_participants).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=lr)
    
    print('Training GRU...')
    gru = training(dataset_train=dataset_training, dataset_test=dataset_test, gru=gru, optimizer=optimizer, epochs=epochs, device=device)
    print('Training GRU done!')
    
    # save GRU model
    torch.save(gru.state_dict(), path_save_model)
    print('Saved GRU to ' + path_save_model)
    
if __name__=='__main__':
    
    # dataset_name = 'eckstein2022'
    # split_ratio = 0.8
    
    # dataset_name = 'eckstein2024'
    # split_ratio = [1, 3]
    
    dataset_name = 'dezfouli2019'
    split_ratio = [3, 6, 9]
    
    # dataset_name = 'gershmanB2018'
    # split_ratio = [4, 8, 12, 16]
    
    path_save_model = f'weinhardt2025/params/{dataset_name}/gru_{dataset_name}.pkl'
    path_data = f'weinhardt2025/data/{dataset_name}/{dataset_name}.csv'
    epochs = 4000
    lr = 1e-2
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    main(path_save_model=path_save_model, path_data=path_data, epochs=epochs, lr=lr, split_ratio=split_ratio, device=device)