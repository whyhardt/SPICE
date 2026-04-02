from typing import Union


class Env:
    
    def __init__(self, n_actions, n_items,):
        self.n_actions = n_actions
        self.n_items = n_items
        self.observation = None
        
    
    def step(self, action: int) -> tuple[dict, bool]:
        
        terminated = True
        
        # process the action
        # generate new observation
        
        return self.observation, terminated

    def reset(self, *args, **kwargs) -> dict:
        
        # init observation space
        # return first observation
        
        return self.observation