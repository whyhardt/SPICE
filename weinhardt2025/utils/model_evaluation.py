import numpy as np

    
def log_likelihood(data: np.ndarray, probs: np.ndarray, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1 
    
    # Sum over all data points
    # return np.sum(np.sum(data * np.log(probs), axis=-1), axis=axis) / normalization
    # Ensure probabilities are within a valid range to prevent log(0)
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1 - epsilon)
    
    # Calculate log-likelihood for each observation
    log_likelihoods = data * np.log(probs)# + (1 - data) * np.log(1 - probs)
    # log_likelihoods = data * np.log(probs)
    # log_likelihoods = np.sum(data * np.log(probs), axis=-1)
    
    # Sum log-likelihoods over all observations
    return np.sum(log_likelihoods)

def bayesian_information_criterion(data: np.ndarray, probs: np.ndarray, n_parameters: int, ll: np.ndarray = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if ll is None:
        ll = log_likelihood(data=data, probs=probs)
    
    n_samples = (data[:, 0] != -1).sum()
    return -2 * ll + n_parameters * np.log(n_samples)

def akaike_information_criterion(data: np.ndarray, probs: np.ndarray, n_parameters: int, ll: np.ndarray = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if ll is None:
        ll = log_likelihood(data=data, probs=probs)
    
    return -2 * ll + 2 * n_parameters

def get_scores(data: np.ndarray, probs: np.ndarray, n_parameters: int, **kwargs) -> float:
        ll = log_likelihood(data=data, probs=probs)
        bic = bayesian_information_criterion(data=data, probs=probs, n_parameters=n_parameters, ll=ll)
        aic = akaike_information_criterion(data=data, probs=probs, n_parameters=n_parameters, ll=ll)
        return -ll, aic, bic
