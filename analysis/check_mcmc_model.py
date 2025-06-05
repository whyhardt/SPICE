import pickle
import arviz as az
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from spice.benchmarking.hierarchical_bayes_numpyro import rl_model

# models = ['ApBr', 'ApBrAcfpBcf', 'ApBrAcfpBcfBch', 'ApAnBrBch', 'ApAnBrAcfpAcfnBcfBch', 'ApAnBrBcfBch']
path_model = 'params/eckstein2022/mcmc_eckstein2022_ApBrAcfpBcfBch.nc' 
# path_model = 'params/eckstein2022/mcmc_eckstein2022_ApBrAcfpBcf.nc'

# setup mcmc agent
with open(path_model, 'rb') as file:
    mcmc = pickle.load(file)

posterior_samples = mcmc.get_samples()
idata = az.from_numpyro(mcmc, log_likelihood=False)
# idata = az.from_dict(posterior_samples)

# az.plot_trace(idata)
# plt.show()

summary = az.summary(idata, var_names=None)
print(summary['r_hat'])

# Save to a CSV file
summary.to_csv("summary.csv")