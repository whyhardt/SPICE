import os
import pandas as pd
import numpy as np

base_file = 'benchmarking/results/results_sugawara.csv'
# base_file = 'benchmarking/results/results_eckstein.csv'

file = base_file.split('/')[-1].split('.')[0]
directory_list = base_file.split('/')[:-1]
directory = ''
for p in directory_list:
    directory += p + '/'

df_dict = {}
df_sindy = None
for f in os.listdir(directory):
    if file in f:
        model = f.split('_')[-1].split('.')[0]
        df_dict[model] = pd.read_csv(directory + f, index_col=0)
        if 'sindy' in f:
            # get sindy data as reference which sessions to ignore
            df_sindy = pd.read_csv(directory + f)

index_ignore = None
if df_sindy is not None:
    index_ignore = np.where(df_sindy['NLL'].isna().values.astype(int) == 1)[0]
    index_valid = (1-df_sindy['NLL'].isna()).astype(bool)
    
models = list(df_dict.keys())
nll, aic, bic = np.zeros(len(df_dict[model])), np.zeros(len(df_dict[model])), np.zeros(len(df_dict[model]))
for i in range(len(df_dict[model])):
    if index_ignore is not None and i in index_ignore:
        min_nll, min_aic, min_bic = 0, 0, 0
    else:
        min_nll, min_aic, min_bic = np.inf, np.inf, np.inf
        for j, m in enumerate(models):
            if m not in ['sindy', 'rnn']:
                if df_dict[m]['NLL'].iloc[i] < min_nll:
                    min_nll = df_dict[m]['NLL'].iloc[i]
                if df_dict[m]['AIC'].iloc[i] < min_aic:
                    min_aic = df_dict[m]['AIC'].iloc[i]
                if df_dict[m]['BIC'].iloc[i] < min_bic:
                    min_bic = df_dict[m]['BIC'].iloc[i]
        nll[i] += min_nll
        aic[i] += min_aic
        bic[i] += min_bic

nll_all, aic_all, bic_all = np.zeros((len(models))), np.zeros((len(models))), np.zeros((len(models)))
for j, m in enumerate(models):
    nll_all[j] = np.sum(df_dict[m]['NLL'].values[index_valid])
    aic_all[j] = np.sum(df_dict[m]['AIC'].values[index_valid])
    bic_all[j] = np.sum(df_dict[m]['BIC'].values[index_valid])

df_result = pd.DataFrame(
    data=np.stack((np.concatenate((nll.sum(keepdims=True), nll_all)), np.concatenate((aic.sum(keepdims=True), aic_all)), np.concatenate((bic.sum(keepdims=True), bic_all)))).T,
    columns=('NLL', 'AIC', 'BIC'),
    index=['Best'] + models,
)

print(df_result)

# print('Loaded file: ' + file)
# print('Minimum values per participant per model:')
# print(f'NLL = {np.sum(nll)} --- AIC = {np.sum(aic)} --- BIC = {np.sum(bic)}')