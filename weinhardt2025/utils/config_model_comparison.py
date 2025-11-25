from typing import List, Optional

from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN

class ConfigModelComparison:
    
    def __init__(
        self,
        study: str = None,
        rnn_class: BaseRNN = None,
        spice_config: SpiceConfig = None,
        file_spice: str = None,
        file_benchmark: str = None,
        file_baseline: str = None,
        file_lstm: str = None,
        additional_inputs: List[str] = None,
        n_parameters_baseline: int = None,
        n_parameters_benchmark: int = None,
        train_ratio_time: float = None,
        test_sessions: List[int] = None,
        setup_agent_benchmark: callable = None,
        rl_model: callable = None,
        ):
        
        self.study = study
        self.rnn_class = rnn_class
        self.spice_config = spice_config
        self.file_spice = file_spice
        self.file_benchmark = file_benchmark
        self.file_baseline = file_baseline
        self.file_lstm = file_lstm
        self.additional_inputs = additional_inputs
        self.n_parameters_baseline = n_parameters_baseline
        self.n_parameters_benchmark = n_parameters_benchmark
        self.train_ratio_time = train_ratio_time
        self.test_sessions = test_sessions
        self.setup_agent_benchmark = setup_agent_benchmark
        self.rl_model = rl_model