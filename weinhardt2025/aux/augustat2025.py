from spice import SpiceConfig

spice_config = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
        'value_reward_not_displayed': [],
    },
    
    memory_state={
        'value_reward': 0.0,
        },
)

import torch
from spice import BaseRNN


class SPICERNN(BaseRNN):

    def __init__(self, spice_config, n_items, n_actions, n_participants, **kwargs):
        super().__init__(spice_config=spice_config, n_items=n_items, n_actions=n_actions, n_participants=n_participants, **kwargs)
        
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=self.embedding_size)

        self.setup_module(key_module='value_reward_chosen', input_size=1+self.embedding_size)
        self.setup_module(key_module='value_reward_not_chosen', input_size=self.embedding_size)
        self.setup_module(key_module='value_reward_not_displayed', input_size=self.embedding_size)

    def forward(self, inputs, prev_state, batch_first=False):

        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        # Get shown items (raw indices) - these are time-shifted, so they refer to the NEXT trial
        shown_at_0_current = spice_signals.additional_inputs[..., 0].long()
        shown_at_1_current = spice_signals.additional_inputs[..., 1].long()
        shown_at_0_next = spice_signals.additional_inputs[..., 2].long()
        shown_at_1_next = spice_signals.additional_inputs[..., 3].long()

        participant_embeddings = self.participant_embedding(spice_signals.participant_ids)

        for timestep in spice_signals.timesteps:

            # Transform input data from action space to item space

            # Determine which action was chosen
            action_idx = spice_signals.actions[timestep].argmax(dim=-1)

            # Map to item indices using current trial's shown items
            item_chosen_idx = torch.where(action_idx == 0, shown_at_0_current[timestep], shown_at_1_current[timestep])
            item_not_chosen_idx = torch.where(action_idx == 1, shown_at_0_current[timestep], shown_at_1_current[timestep])

            # Create one-hot masks
            item_chosen_onehot = torch.nn.functional.one_hot(item_chosen_idx, num_classes=self.n_items).float()
            item_not_chosen_onehot = torch.nn.functional.one_hot(item_not_chosen_idx, num_classes=self.n_items).float()
            item_not_displayed_onehot = 1 - (item_chosen_onehot + item_not_chosen_onehot)

            # Map rewards from action space to item space
            reward_action = spice_signals.rewards[timestep, :]  # shape: (batch, n_actions)

            # Create reward tensor in item space (batch, n_items)
            reward_item = torch.zeros(reward_action.shape[0], self.n_items, device=reward_action.device)

            # Scatter rewards to the corresponding items:
            # Item at shown_at_0_current gets reward for action 0
            # Item at shown_at_1_current gets reward for action 1
            reward_item.scatter_(1, shown_at_0_current[timestep].unsqueeze(-1), reward_action[:, 0].unsqueeze(-1))
            reward_item.scatter_(1, shown_at_1_current[timestep].unsqueeze(-1), reward_action[:, 1].unsqueeze(-1))
            
            # Update chosen
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=item_chosen_onehot,
                inputs=reward_item,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # Update not chosen
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=item_not_chosen_onehot,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # Update not displayed
            self.call_module(
                key_module='value_reward_not_displayed',
                key_state='value_reward',
                action_mask=item_not_displayed_onehot,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # Transform values from item space to action space for NEXT trial (for prediction)
            # Use the time-shifted items (next trial's items)
            value_at_0 = torch.gather(self.state['value_reward'], 1, shown_at_0_next[timestep].unsqueeze(-1))
            value_at_1 = torch.gather(self.state['value_reward'], 1, shown_at_1_next[timestep].unsqueeze(-1))

            # log action values
            spice_signals.logits[timestep] = torch.concat([value_at_0, value_at_1], dim=-1)

        spice_signals = self.post_forward_pass(spice_signals, batch_first)

        return spice_signals.logits, self.get_state()


# GRU specifications:
# 1. In order to function seamlessly with the rest of the framework the GRU has to output (logits, hidden_state)
# 2. Because of the task type (more items to remember than displayed for action; n_items > n_actions):
#   2.1 The GRU computes the action values for all items at each timestep (chosen, not chosen but displayed, and not displayed)
#   2.2 To facilitate selection only among the shown options, the GRU has to map the values from the item space (all values) to the action space (only displayed ones)

class GRU(torch.nn.Module):
    
    def __init__(self, n_actions, n_items, additional_inputs: int = 0, hidden_size: int = 32, **kwargs):
        super().__init__()
        
        self.gru_features = hidden_size
        self.n_items = n_items
        self.n_actions = n_actions
        self.additional_inputs = additional_inputs

        self.linear_in = torch.nn.Linear(in_features=n_actions+1+additional_inputs, out_features=hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear_out = torch.nn.Linear(in_features=hidden_size, out_features=n_items)
        
    def forward(self, inputs, state=None):
        
        actions = inputs[..., :self.n_actions]
        rewards = inputs[..., self.n_actions:2*self.n_actions].nan_to_num(0).sum(dim=-1, keepdims=True)
        item_pairs = inputs[..., self.n_actions*2:self.n_actions*2+self.additional_inputs]
        inputs = torch.concat((actions, rewards, item_pairs), dim=-1)
        
        # Get item pairs
        # item_pairs = inputs[..., 3*self.n_actions:3*self.n_actions+2]
        
        if state is not None and len(inputs.shape) == 3:
            state = state.reshape(1, 1, self.gru_features)
        
        y = self.linear_in(inputs.nan_to_num(0))
        y = self.dropout(y)
        y, state = self.gru(y, state)
        y = self.dropout(y)
        y = self.linear_out(y)
        
        # map values from item space into action space to determine the next action based on the shown options
        item1_values = torch.gather(y, 2, item_pairs[..., 0].unsqueeze(-1).nan_to_num(0).long())
        item2_values = torch.gather(y, 2, item_pairs[..., 1].unsqueeze(-1).nan_to_num(0).long())
        y = torch.cat([item1_values, item2_values], dim=-1)
        
        return y, state
    
    
if __name__=='__main__':
    
    from spice import SpiceDataset, convert_dataset

    # Load your data
    dataset = convert_dataset(
        file = 'weinhardt2025/data/augustat2025/augustat2025.csv',
        df_participant_id='participant_id',
        df_choice='choice',
        df_reward='reward',
        additional_inputs=['shown_at_0', 'shown_at_1'],
        timeshift_additional_inputs=False,
        )
    
    n_actions = dataset.ys.shape[-1]
    n_items = dataset.xs[..., 2*n_actions+1].nan_to_num(-1).max().int() + 1
    # in order to set up the participant embedding we have to compute the number of unique participants in our data 
    # to get the number of participants n_participants we do:
    n_participants = len(dataset.xs[..., -1].unique())
    
    # add shown_at_i_next (alongside shown_at_i)
    shown_at_i_next = dataset.xs[:, 1:, n_actions*2:n_actions*2+2]
    xs = torch.concat((dataset.xs[:, :-1, :n_actions*2+2], shown_at_i_next, dataset.xs[:, :-1, -3:]), dim=-1)
    ys = dataset.ys[:, :-1]
    dataset = SpiceDataset(xs, ys)

    # TODO: split into training and test data
    dataset_train, dataset_test = dataset, dataset
    
    from spice import SpiceEstimator

    path_spice = 'weinhardt2025/params/augustat2025/spice_augustat2025.pkl'

    estimator = SpiceEstimator(
            # model paramaeters
            rnn_class=SPICERNN,
            spice_config=spice_config,
            n_actions=n_actions,
            n_items=n_items,
            n_participants=n_participants,
            
            # rnn training parameters
            epochs=1000,
            warmup_steps=250,
            learning_rate=0.01,
            
            # sindy fitting parameters
            sindy_weight=0,#0.1,
            sindy_threshold=0.05,
            sindy_threshold_frequency=1,
            sindy_threshold_terms=1,
            sindy_cutoff_patience=100,
            sindy_epochs=1000,
            sindy_alpha=0.0001,
            sindy_library_polynomial_degree=2,
            sindy_ensemble_size=1,
            
            # additional generalization parameters
            bagging=True,
            scheduler=True,
            
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            verbose=True,
            save_path_spice=path_spice,
        )
    
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    
    # import sys

    # sys.path.append('../..')
    # from weinhardt2025.benchmarking.benchmarking_gru import training, setup_agent_gru

    # path_gru = 'weinhardt2025/params/augustat2025/gru_augustat2025.pkl'
    
    # epochs = 10000

    # gru = GRU(n_actions=n_actions, n_items=n_items, additional_inputs=2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    
    # gru = training(
    #     gru=gru,
    #     optimizer=optimizer,
    #     dataset_train=dataset_train,
    #     dataset_test=dataset_test,
    #     epochs=epochs,
    #     )

    # torch.save(gru.state_dict(), path_gru)
    # print("Trained GRU parameters saved to " + path_gru)
    
    # gru_agent = setup_agent_gru(path_gru, gru)
    
    # import matplotlib.pyplot as plt
    # from spice import plot_session
    
    # # plotting
    # participant_id = 7

    # # estimator.print_spice_model(participant_id)

    # agents = {
    #     # add baseline agent here
    #     'rnn': estimator.rnn_agent,
    #     'spice': estimator.spice_agent,
    #     # 'baseline': baseline_agent,
    #     'gru': gru_agent,
    # }

    # fig, axs = plot_session(agents, dataset.xs[participant_id], signals_to_plot=[])
    # plt.show()