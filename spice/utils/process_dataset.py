import pandas as pd
import numpy as np


path = 'data/sugawara2021_143_raw.csv'
data = pd.read_csv(path)

session_id = 'subjectid'
reward_id = 'reward'
choice_id = 'key'
rt_id = 'rt'

mapping_key = {
    0: -1,
    33: 0,
    36: 1,
}

mapping_reward = {
    -10: 0,
    10: 1,
}

rewards = data[reward_id].values
choices = data[choice_id].values
rts = data[rt_id].values
sessions = data[session_id].values

for m in mapping_key:
    choices[choices == m] = mapping_key[m]

for m in mapping_reward:
    rewards[rewards == m] = mapping_reward[m]
    
rewards = rewards[choices != -1]
rts = rts[choices != -1]
sessions = sessions[choices != -1]
choices = choices[choices != -1]

data = pd.DataFrame(np.stack((sessions, choices, rewards, rts), axis=-1), columns=('session', 'choice', 'reward', 'rt'))

print(data)
data.to_csv('data/sugawara2021_143_processed.csv', index=False)