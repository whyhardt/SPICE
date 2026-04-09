import torch

from spice import SpiceDataset, csv_to_dataset, split_data_along_sessiondim, split_data_along_timedim, cross_entropy_loss


class GRUModel(torch.nn.Module):
    
    def __init__(
        self, 
        n_actions, 
        additional_inputs: int = 0, 
        hidden_size: int = 16, 
        n_reward_features: int = None, 
        dropout: float = 0.1, 
        n_participants: int = 1, 
        n_experiments: int = 1, 
        embedding_size: int = 16, 
        **kwargs,
        ):
        super().__init__()
        
        self.gru_features = hidden_size
        self.n_actions = n_actions
        self.additional_inputs = additional_inputs
        self.n_reward_features = n_actions if n_reward_features is None else n_reward_features
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        self.participant_embedding_size = embedding_size if n_participants > 1 else 0
        self.experiment_embedding_size = embedding_size if n_experiments > 1 else 0
        
        self.participant_embedding = torch.nn.Embedding(num_embeddings=n_participants, embedding_dim=embedding_size)
        self.experiment_embedding = torch.nn.Embedding(num_embeddings=n_experiments, embedding_dim=embedding_size)
        
        self.linear_in = torch.nn.Linear(in_features=self.n_actions+self.n_reward_features+self.additional_inputs+self.participant_embedding_size+self.experiment_embedding_size, out_features=hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True,)
        self.linear_out = torch.nn.Linear(in_features=hidden_size, out_features=n_actions)
        
    def forward(self, inputs: torch.Tensor, state: torch.Tensor = None):
        
        inputs = inputs.nan_to_num(0.)
        actions = inputs[..., :self.n_actions]
        rewards = inputs[..., self.n_actions:self.n_actions+self.n_reward_features]
        additional_inputs = inputs[..., self.n_actions+self.n_reward_features:self.n_actions+self.n_reward_features+self.additional_inputs]
        xs = torch.concat((actions, rewards, additional_inputs), dim=-1)

        if self.n_experiments > 1:
            experiment_embedding = self.experiment_embedding[inputs[..., -2].long()]
            xs = torch.concat((xs, experiment_embedding), dim=-1)
        
        if self.n_participants > 1:
            participant_embedding = self.participant_embedding(inputs[..., -1].long())
            xs = torch.concat((xs, participant_embedding), dim=-1)
        
        if state is not None and len(inputs.shape) == 3:
            state = state.reshape(1, 1, self.gru_features)
        
        y = self.linear_in(xs[:, :, 0])
        y = self.dropout(y)
        y, state = self.gru(y, state)
        y = self.dropout(y)
        y = self.linear_out(y).unsqueeze(2)
        return y, state


def training(
    model: GRUModel, 
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
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # random_indexes = torch.randint(dataset_train.xs.shape[0], (1, dataset_train.xs.shape[0]))[0]
        
        # xs_train = dataset_train.xs[random_indexes].to(device)
        # ys_train = dataset_train.ys[random_indexes].to(device)
        # xs_test = dataset_test.xs.to(device)
        # ys_test = dataset_test.ys.to(device)
        
        for batch in dataloader:
            
            xs, ys = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            logits, _ = model(xs)
            
            # Reshape for loss computation
            nan_mask = ~torch.isnan(xs[..., :model.n_actions].sum(dim=-1))
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
            model.eval()
            with torch.no_grad():
                logits_test, _ = model(dataset_test.xs.to(device))
                nan_mask = ~torch.isnan(dataset_test.xs[..., :model.n_actions].sum(dim=-1))
                logits_test = logits_test[nan_mask].to(device)
                # labels_test = torch.argmax(dataset_test.ys[..., :n_actions].to(device).reshape(-1, n_actions)[nan_mask], dim=-1).reshape(-1).long()
                labels_test = dataset_test.ys[nan_mask].to(device)
                loss_test = criterion(logits_test, labels_test)
            model.train()
            
            msg += f"; L(Test): {loss_test.item()}"
        
        print(msg)
        
    return model


def main(path_save_model:str, path_data: str, epochs: int, lr: float, split_ratio: float, device=torch.device('cpu')):
    
    dataset = csv_to_dataset(path_data)
    dataset.normalize_rewards()
    if isinstance(split_ratio, float):
        dataset_training, dataset_test = split_data_along_timedim(dataset, split_ratio=split_ratio)
    else:
        dataset_training, dataset_test = split_data_along_sessiondim(dataset, test_sessions=split_ratio)
    
    n_actions = dataset_training.ys.shape[-1]
    n_participants = len(dataset_training.xs[..., -1].unique())
    
    gru = GRUModel(input_size=n_actions+1, n_actions=n_actions, n_participants=n_participants).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=lr)
    
    print('Training GRU...')
    gru = training(dataset_train=dataset_training, dataset_test=dataset_test, model=gru, optimizer=optimizer, epochs=epochs, device=device)
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
    
    path_save_model = f'weinhardt2026/params/{dataset_name}/gru_{dataset_name}.pkl'
    path_data = f'weinhardt2026/data/{dataset_name}/{dataset_name}.csv'
    epochs = 4000
    lr = 1e-2
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    main(path_save_model=path_save_model, path_data=path_data, epochs=epochs, lr=lr, split_ratio=split_ratio, device=device)