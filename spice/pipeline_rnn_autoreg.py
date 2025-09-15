#@title Import libraries
import sys
import os

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import argparse
from typing import List

# warnings.filterwarnings("ignore")

# RL libraries
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '')))
from spice.resources import rnn, rnn_training, bandits, rnn_utils, rnn_training_awd, rnn_training_imaml
from spice.utils import convert_dataset, plotting

def main(
  checkpoint = False,
  model: str = None,
  data: str = None,
  class_rnn: type = None,

  # rnn parameters
  hidden_size = 8,
  embedding_size = 32,
  dropout = 0.5,

  # meta-optimization parameters
  metaopt_type: str = 'awd', # 'awd' or 'iMAML'
  # AWD parameters
  lambda_awd: float = 0.022,  # Default from paper experiments
  # iMAML parameters
  meta_update_interval: int = 50,
  inner_steps: int = 3,
  outer_lr: float = 1e-2,
  hypergradient_steps: int = 3,
  initial_reg_param: float = 1e-1,

  # data and training parameters
  epochs = 128,
  train_test_ratio = 1.,
  l2_weight_decay=0,
  bagging = False,
  sequence_length = -1, # -1 for full sequence
  n_steps = -1,  # -1 for full sequence
  batch_size = -1,  # -1 for one batch per epoch
  learning_rate = 1e-2,
  convergence_threshold = 0,
  scheduler = False,
  additional_inputs_data: List[str] = None,

  # ground truth parameters
  n_trials = 200,
  n_sessions = 256,
  beta_reward = 3.,
  alpha_reward = 0.25,
  alpha_penalty = -1.,
  alpha_counterfactual = 0.,
  beta_choice = 0.,
  alpha_choice = 0.,
  forget_rate = 0.,
  confirmation_bias = 0.,
  parameter_variance = 0.,
  reward_prediction_error: Callable = None,
  
  # environment parameters
  n_actions = 2,
  sigma = 0.1,
  counterfactual = False,
  
  analysis: bool = False,
  participant_id: int = 0,
  save_checkpoints: bool = False,
  ):
  
  # print cuda devices available
  print(f'Cuda available: {torch.cuda.is_available()}')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')
  
  if not os.path.exists('params'):
    os.makedirs('params')

  if participant_id is None:
    participant_id = 0
  
  
  dataset_test = None
  agent = None
  if data is None:
    print('No path to dataset provided.')
    
    # setup
    environment = bandits.BanditsDrift(sigma=sigma, n_actions=n_actions, counterfactual=counterfactual)
    # environment = bandits.EnvironmentBanditsSwitch(sigma, counterfactual=counterfactual)
    agent = bandits.AgentQ(
      n_actions=n_actions, 
      beta_reward=beta_reward, 
      alpha_reward=alpha_reward, 
      alpha_penalty=alpha_penalty, 
      forget_rate=forget_rate, 
      confirmation_bias=confirmation_bias, 
      beta_choice=beta_choice, 
      alpha_choice=alpha_choice,
      alpha_counterfactual_reward=alpha_counterfactual,
      alpha_counterfactual_penalty=alpha_counterfactual, 
      parameter_variance=parameter_variance,
      )
    if reward_prediction_error is not None:
      agent.set_reward_prediction_error(reward_prediction_error)
    print('Setup of the environment and agent complete.')
    
    print('Generating the synthetic dataset...')
    dataset, _, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials=n_trials,
        n_sessions=n_sessions,
        sample_parameters=parameter_variance!=0,
        sequence_length=sequence_length,
        device=device,
        )
    
    # set participant ids to 0
    dataset.xs[..., -1] = 0.
    
    if train_test_ratio == 1:
      dataset_test, _, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials=256,
        n_sessions=n_sessions,
        sample_parameters=parameter_variance!=0,
        device=device)
      
      dataset_test.xs[..., -1] = 0.

    print('Generation of dataset complete.')
  else:
    dataset, _, df, _ = convert_dataset.convert_dataset(data, sequence_length=sequence_length, device=device, additional_inputs=additional_inputs_data)
    # dataset_test = rnn_utils.DatasetRNN(dataset.xs, dataset.ys)
    
    # check if groundtruth parameters in data - only applicable to generated data with e.g. utils/create_dataset.py
    if 'mean_beta_reward' in df.columns:
      # get parameters from dataset
      agent = bandits.AgentQ(
        beta_reward = df['beta_reward'].values[(df['session']==participant_id).values][0],
        alpha_reward = df['alpha_reward'].values[(df['session']==participant_id).values][0],
        alpha_penalty = df['alpha_penalty'].values[(df['session']==participant_id).values][0],
        confirmation_bias = df['confirmation_bias'].values[(df['session']==participant_id).values][0],
        forget_rate = df['forget_rate'].values[(df['session']==participant_id).values][0],
        beta_choice = df['beta_choice'].values[(df['session']==participant_id).values][0],
        alpha_choice = df['alpha_choice'].values[(df['session']==participant_id).values][0],
      )
      
  n_participants = len(dataset.xs[..., -1].unique())

  if isinstance(train_test_ratio, float):
    if train_test_ratio < 1:
      dataset_train, dataset_test = rnn_utils.split_data_along_timedim(dataset, train_test_ratio)
      dataset_train = bandits.DatasetRNN(dataset_train.xs, dataset_train.ys, sequence_length=sequence_length, device=device)
      dataset_val, dataset_test = rnn_utils.split_data_along_timedim(dataset_test, 0.5) # Split test set for val set
    else:
      dataset_train = bandits.DatasetRNN(dataset.xs, dataset.ys, sequence_length=sequence_length, device=device)
  elif isinstance(train_test_ratio, list):
    val_test_split = len(train_test_ratio) // 2
    # train_test_ratio are all the test sessions
    val_sessions = train_test_ratio[:val_test_split]
    test_sessions = train_test_ratio[val_test_split:]

    dataset_train, _ = rnn_utils.split_data_along_sessiondim(dataset=dataset, list_test_sessions=train_test_ratio, device=device)
    _, dataset_val = rnn_utils.split_data_along_sessiondim(dataset=dataset, list_test_sessions=val_sessions, device=device)
    _, dataset_test = rnn_utils.split_data_along_sessiondim(dataset=dataset, list_test_sessions=test_sessions, device=device)
  elif train_test_ratio is None:
    dataset_train, dataset_test = dataset, dataset
  elif train_test_ratio is None:
    dataset_train, dataset_test = dataset, dataset
  else:
    raise TypeError("train_test_raio must be either a float number or a list of integers containing the session/block ids which should be used as test sessions/blocks")

  if data is None and model is None:
    params_path = rnn_utils.parameter_file_naming(
      'params/params', 
      alpha_reward=alpha_reward, 
      beta_reward=beta_reward, 
      alpha_counterfactual=alpha_counterfactual,
      forget_rate=forget_rate, 
      beta_choice=beta_choice,
      alpha_choice=alpha_choice,
      alpha_penalty=alpha_penalty,
      confirmation_bias=confirmation_bias, 
      variance=parameter_variance, 
      verbose=True,
      )
  elif data is not None and model is None:
    params_path = 'params/params_' + data.split('/')[-1].replace('.csv', '.pkl')
  else:
    params_path = '' + model

  # define model
  if class_rnn is None:
    class_rnn = rnn.RLRNN
  model = class_rnn(
      n_actions=n_actions, 
      hidden_size=hidden_size, 
      embedding_size=embedding_size,
      dropout=dropout,
      n_participants=n_participants,
      ).to(device)
  
  optimizer_rnn = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight_decay)
  
  print('Setup of the RNN model complete.')

  if checkpoint:
      model, optimizer_rnn = rnn_utils.load_checkpoint(params_path, model, optimizer_rnn)
      print('Loaded model parameters.')

  loss_test = None
  if epochs > 0:
    start_time = time.time()
    
    #Fit the RNN
    print('Training the RNN...')
    if metaopt_type=='awd':
      print('Using Adaptive Weight Decay (AWD) for meta-optimization.')
      model, optimizer_rnn, histories = rnn_training_awd.fit_with_metaopt(
        model=model,
        model_optimizer=optimizer_rnn,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        epochs=epochs,
        batch_size=batch_size,
        convergence_threshold=convergence_threshold,
        bagging=bagging,
        n_steps=n_steps,
        scheduler=scheduler,
        path_save_checkpoints=params_path if save_checkpoints else None,

        lambda_awd=lambda_awd,
      )
    elif metaopt_type=='iMAML':
      print('Using Implicit MAML (iMAML) for meta-optimization.')
      model, optimizer_rnn, histories = rnn_training_imaml.fit_with_metaopt(
        model=model,
        model_optimizer=optimizer_rnn,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        epochs=epochs,
        batch_size=batch_size,
        convergence_threshold=convergence_threshold,
        bagging=bagging,
        n_steps=n_steps,
        scheduler=scheduler,
        path_save_checkpoints=params_path if save_checkpoints else None,

        meta_update_interval=meta_update_interval,
        inner_steps=inner_steps,
        hypergradient_steps=hypergradient_steps,
        outer_lr=outer_lr,
        initial_reg_param=initial_reg_param,
    )
        
    else:
      raise ValueError("metaopt_type must be either 'awd' or 'iMAML'")

    # save trained parameters
    state_dict = {'model': model.state_dict(), 'optimizer': optimizer_rnn.state_dict()}
    
    print('Training finished.')
    torch.save(state_dict, params_path)
    print(f'Saved RNN parameters to file {params_path}.')
    print(f'Training took {time.time() - start_time:.2f} seconds.')
  
  # validate model
  str_data = 'test' if dataset_test is not None else 'train'
  print(f'\nTesting the trained RNN on the', str_data, 'dataset...')
  model.eval()
  with torch.no_grad():
    _, _, loss_test = rnn_training.fit_model(
        model=model,
        dataset_train=dataset_test if dataset_test is not None else dataset_train,
    )
  
  # -----------------------------------------------------------
  # Analysis
  # -----------------------------------------------------------
  
  if analysis:
    # print(f'Betas of model: {(model._beta_reward.item(), model._beta_choice.item())}')
    # Synthesize a dataset using the fitted network
    agent_rnn = bandits.AgentNetwork(model_rnn=model, n_actions=n_actions)
    
    # get analysis plot
    if agent is not None:
      agents = {'groundtruth': agent, 'rnn': agent_rnn}
    else:
      agents = {'rnn': agent_rnn}

    # fig, axs = plotting.plot_session(agents, dataset_test.xs[participant_id] if dataset_test is not None else dataset_train.xs[participant_id])
    fig, axs = plotting.plot_session(agents, dataset.xs[participant_id])
    
    title_ground_truth = ''
    if agent is not None:
      title_ground_truth += r'GT: $\beta_{reward}=$'+str(np.round(agent._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent._beta_choice, 2))
    betas = agent_rnn.get_betas()
    title_rnn = r'RNN: $\beta_{reward}=$'+str(np.round(betas['x_value_reward'], 2)) + r'; $\beta_{choice}=$'+str(np.round(betas['x_value_choice'], 2))
    fig.suptitle(title_ground_truth + '\n' + title_rnn)
    plt.show()
    
  # Return histories as well
  return model, loss_test, histories


if __name__=='__main__':
  
  parser = argparse.ArgumentParser(description='Trains a SPICE-RNN on behavioral data to uncover the underlying Q-Values via different cognitive modules.')
  
  
  parser.add_argument('--checkpoint', action='store_true', help='Whether to load a checkpoint')
  parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
  parser.add_argument('--data', type=str, default=None, help='Path to dataset')
  
  # RNN parameters
  parser.add_argument('--hidden_size', type=int, default=8, help='Hidden size of the RNN')
  parser.add_argument('--embedding_size', type=int, default=32, help='Participant embedding size of the RNN')
  parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

  # Meta-optimization parameters
  parser.add_argument('--metaopt_type', type=str, default='awd', choices=['awd','iMAML'], help='Meta-optimization method')
  parser.add_argument('--lambda_awd', type=float, default=0.022, help='AWD regularization strength, only used if metaopt_type==\'awd\'')
  parser.add_argument('--meta_update_interval', type=int, default=50, help='Number of epochs between meta-updates, only used if metaopt_type==\'iMAML\'')
  parser.add_argument('--inner_steps', type=int, default=3, help='Number of training steps into the future, only used if metaopt_type==\'iMAML\'')
  parser.add_argument('--outer_lr', type=float, default=1e-2, help='Learning rate for meta-parameters, only used if metaopt_type==\'iMAML\'')
  parser.add_argument('--hypergradient_steps', type=int, default=3, help='Number of steps used by Conjugate Gradient calculation, only used if metaopt_type==\'iMAML\'')
  parser.add_argument('--initial_reg_param', type=float, default=1e-1, help='Initial regularization parameter lambda, only used if metaopt_type==\'iMAML\'')

  # data and training parameters
  parser.add_argument('--scheduler', action='store_true', help='Whether to use a learning rate scheduler during training')
  parser.add_argument('--bagging', action='store_true', help='Whether to use bagging')
  parser.add_argument('--epochs', type=int, default=8192, help='Number of epochs for training')
  parser.add_argument('--n_steps', type=int, default=-1, help='Number of recurrent steps per training call; -1: Use whole sequence at once;')
  parser.add_argument('--batch_size', type=int, default=-1, help='Batch size; -1: Use whole dataset at once;')
  parser.add_argument('--sequence_length', type=int, default=-1, help='Length of training sequences; -1: Use whole sequence at once;')
  parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate of the RNN')
  parser.add_argument('--l2_weight_decay', type=float, default=0, help='Learning rate of the RNN')
  parser.add_argument('--convergence_threshold', type=float, default=0, help='Convergence threshold to early-stop training')
  parser.add_argument('--train_test_ratio', type=str, default="1.0", help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')
  
  # Ground truth parameters
  parser.add_argument('--n_trials', type=int, default=200, help='Number of trials per session')
  parser.add_argument('--n_sessions', type=int, default=256, help='Number of sessions')
  parser.add_argument('--beta_reward', type=float, default=3, help='Beta parameter for the Q-learning update rule')
  parser.add_argument('--alpha_reward', type=float, default=0.25, help='Alpha parameter for the Q-learning update rule')
  parser.add_argument('--alpha_penalty', type=float, default=-1., help='Learning rate for negative outcomes; if -1: same as alpha')
  parser.add_argument('--forget_rate', type=float, default=0., help='Forget rate')
  parser.add_argument('--beta_choice', type=float, default=0., help='Beta parameter for the Q-learning update rule')
  parser.add_argument('--alpha_choice', type=float, default=1., help='Alpha parameter for the Q-learning update rule')
  parser.add_argument('--alpha_counterfactual', type=float, default=0., help='Alpha parameter for the Q-learning update rule')

  # Environment parameters
  parser.add_argument('--n_actions', type=int, default=2, help='Number of possible actions')
  parser.add_argument('--sigma', type=float, default=0.2, help='Drift rate of the reward probabilities')
  parser.add_argument('--counterfactual', action='store_true', help='Counterfactual experiment with full feedback (for chosen and not chosen options)')

  # Analysis parameters
  parser.add_argument('--analysis', action='store_true', help='Whether to perform visual analysis on one participant (keyword argument: participant_id)')
  parser.add_argument('--participant_id', type=int, default=None, help='Participant ID for visual analysis (keyword argument: analysis)')
  parser.add_argument('--save_checkpoints', action='store_true', help='Save a checkpoint after 2^n training steps with $n\in\mathbb\{N\}$.')
  
  args = parser.parse_args()  
  
  # currently fixed 
  class_rnn = rnn.RLRNN_eckstein2022
  
  # convert train_test_ratio to number of list of numbers
  if ',' in args.train_test_ratio:
    train_test_ratio = [int(x) for x in args.train_test_ratio.split(',')]
  else:
    train_test_ratio = float(args.train_test_ratio)
    
  main(
    checkpoint = args.checkpoint,
    model = args.model,
    data = args.data,
    class_rnn=class_rnn,
    
    # rnn parameters
    hidden_size = args.hidden_size,
    embedding_size = args.embedding_size,
    dropout = args.dropout,

    # meta-optimization parameters
    metaopt_type=args.metaopt_type,
    lambda_awd=args.lambda_awd,
    meta_update_interval=args.meta_update_interval,
    inner_steps=args.inner_steps,
    outer_lr=args.outer_lr,
    hypergradient_steps=args.hypergradient_steps,
    initial_reg_param=args.initial_reg_param,

    # data and training parameters
    epochs = args.epochs,
    train_test_ratio = train_test_ratio,
    n_trials = args.n_trials,
    n_sessions = args.n_sessions,
    bagging = args.bagging,
    sequence_length = args.sequence_length,
    n_steps = args.n_steps,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    convergence_threshold = args.convergence_threshold,
    scheduler = args.scheduler,
    l2_weight_decay=args.l2_weight_decay,
    
    # ground truth parameters
    beta_reward = args.beta_reward,
    alpha_reward = args.alpha_reward,
    alpha_penalty = args.alpha_penalty,
    alpha_counterfactual = args.alpha_counterfactual,
    beta_choice = args.beta_choice,
    alpha_choice = args.alpha_choice,
    forget_rate = args.forget_rate,
    
    # environment parameters
    n_actions = args.n_actions,
    sigma = args.sigma,
    counterfactual = args.counterfactual,
    
    # analysis parameters
    analysis = args.analysis,
    save_checkpoints=args.save_checkpoints,
    participant_id = args.participant_id,
    )
  