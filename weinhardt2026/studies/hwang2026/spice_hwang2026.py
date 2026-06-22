import torch
from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        'module_action': ['action_id1', 'grooming_id1', 'gesture_id1', 'scratch_id1', 'action_id2', 'grooming_id2', 'gesture_id2', 'scratch_id2'],
        'module_grooming': ['action_id1', 'grooming_id1', 'gesture_id1', 'scratch_id1', 'action_id2', 'grooming_id2', 'gesture_id2', 'scratch_id2'],
        'module_gesture': ['action_id1', 'grooming_id1', 'gesture_id1', 'scratch_id1', 'action_id2', 'grooming_id2', 'gesture_id2', 'scratch_id2'],
        # TODO: uncomment when working with scratch data
        # 'module_scratch': ['action_id1', 'grooming_id1', 'gesture_id1', 'scratch_id1', 'action_id2', 'grooming_id2', 'gesture_id2', 'scratch_id2']
        },
    memory_state={
        'values': 0,
        },
    additional_inputs=('SigAct_ID2', 'ID1', 'ID2'),
)


class SpiceModel(BaseModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # participant embedding
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout, target_embedding_size_fusion=8)
        
        # rnn modules
        self.setup_module(key_module='module_action', input_size=8, embedding_size=self.embedding_size*2, dropout=self.dropout)
        self.setup_module(key_module='module_grooming', input_size=8, embedding_size=self.embedding_size*2, dropout=self.dropout)
        self.setup_module(key_module='module_gesture', input_size=8, embedding_size=self.embedding_size*2, dropout=self.dropout)
        # TODO: uncomment when working with scratch data
        # self.setup_module(key_module='module_scratch', input_size=8, embedding_size=self.embedding_size*2, dropout=self.dropout)

    def forward(self, inputs, prev_state=None):
        
        spice_signals = self.init_forward_pass(inputs, prev_state)
        
        # time-invariant participant features
        participant_embeddings_id1 = self.participant_embedding(spice_signals.participant_ids)
        participant_embeddings_id2 = self.participant_embedding(spice_signals.experiment_ids)
        
        # feature extraction
        # id 1
        action_id1 = spice_signals.actions[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        grooming_id1 = spice_signals.actions[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        gesture_id1 = spice_signals.actions[..., 2].unsqueeze(-1).expand_as(spice_signals.actions)
        scratch_id1 = spice_signals.actions[..., 3].unsqueeze(-1).expand_as(spice_signals.actions)
        # id 2
        actions_id2 = torch.eye(self.n_actions, device=self.device)[spice_signals.additional_inputs['SigAct_ID2'].squeeze(-1).int()]  # one hot encoding of actions id2
        action_id2 = actions_id2[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        grooming_id2 = actions_id2[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        gesture_id2 = actions_id2[..., 2].unsqueeze(-1).expand_as(spice_signals.actions)
        scratch_id2 = actions_id2[..., 3].unsqueeze(-1).expand_as(spice_signals.actions)
        
        # action masks
        # TODO: add a zero to the end if using scratch data: (1, 0, 0, 0) -> (1, 0, 0, 0, 0)
        mask_action = torch.tensor((1, 0, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_grooming = torch.tensor((0, 1, 0, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        mask_gesture = torch.tensor((0, 0, 1, 0), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        # TODO: uncomment when working with scratch data
        # mask_scratch = torch.tensor((0, 0, 0, 1), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        # TODO: add a zero to the start if using scratch data: (0, 0, 0, 1) -> (0, 0, 0, 0, 1)
        mask_waiting = torch.tensor((0, 0, 0, 1), device=self.device).reshape(1, 1, 1, 1, -1).expand_as(spice_signals.actions)
        
        for timestep in spice_signals.trials:
            
            # update chosen value
            self.call_module(
                key_module='module_action',
                key_state='values',
                action_mask=mask_action, 
                inputs=(
                    action_id1[timestep],
                    grooming_id1[timestep], 
                    gesture_id1[timestep],
                    scratch_id1[timestep],
                    action_id2[timestep],
                    grooming_id2[timestep], 
                    gesture_id2[timestep],
                    scratch_id2[timestep],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=participant_embeddings_id2,
            )
            
            self.call_module(
                key_module='module_grooming',
                key_state='values',
                action_mask=mask_grooming, 
                inputs=(
                    action_id1[timestep],
                    grooming_id1[timestep], 
                    gesture_id1[timestep],
                    scratch_id1[timestep],
                    action_id2[timestep],
                    grooming_id2[timestep], 
                    gesture_id2[timestep],
                    scratch_id2[timestep],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=participant_embeddings_id2,
            )
            
            self.call_module(
                key_module='module_gesture',
                key_state='values',
                action_mask=mask_gesture, 
                inputs=(
                    action_id1[timestep],
                    grooming_id1[timestep], 
                    gesture_id1[timestep],
                    scratch_id1[timestep],
                    action_id2[timestep],
                    grooming_id2[timestep], 
                    gesture_id2[timestep],
                    scratch_id2[timestep],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings_id1,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=participant_embeddings_id2,
            )
            
            # TODO: uncomment when working with scratch data
            # self.call_module(
            #     key_module='module_scratch',
            #     key_state='values',
            #     action_mask=mask_scratch, 
            #     inputs=(
            #         action_id1[timestep],
            #         grooming_id1[timestep], 
            #         gesture_id1[timestep],
            #         scratch_id1[timestep],
            #         action_id2[timestep],
            #         grooming_id2[timestep], 
            #         gesture_id2[timestep],
            #         scratch_id2[timestep],
            #         ),
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embeddings_id1,
            #     experiment_index=spice_signals.experiment_ids,
            #     experiment_embedding=participant_embeddings_id2,
            # )
            
            spice_signals.logits[timestep] = self.state['values']
                
        spice_signals = self.post_forward_pass(spice_signals)
        
        return spice_signals.logits, self.get_state()
    
    
def cross_entropy_loss_mask_waiting(prediction, target, label_smoothing: float = 0.):
    
    n_actions = target.shape[-1]
    waiting_action = n_actions - 1
    
    prediction = prediction.reshape(-1, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)
    
    non_waiting_mask = target != waiting_action
    if not non_waiting_mask.any():
        return prediction.sum() * 0.
    
    return torch.nn.functional.cross_entropy(
        prediction[non_waiting_mask],
        target[non_waiting_mask],
        label_smoothing=label_smoothing,
    )
