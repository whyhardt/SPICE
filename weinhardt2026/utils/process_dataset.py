import os
import pandas as pd
import numpy as np


path = 'data/raw_data/eckstein2022'

meta_data = 'data/raw_data/eckstein2022/SLCN.csv'

kw_participant_id = 'sID'
kw_age = 'age - years'
kw_gender = 'gender'
kw_reward_id = 'reward'
kw_choice_id = 'selected_box'
kw_correct_choice_id = 'correct_box'
kw_rt_id = 'RT'

columns_out = ['session', 'age', 'gender', 'choice', 'reward', 'correct_choice', 'rt']


# read meta data and map meta data to participant id
meta_data = pd.read_csv(meta_data)

all_data = []
for file in os.listdir(path):

    try:
        data = pd.read_csv(os.path.join(path, file))

        rewards = data[kw_reward_id].values
        choices = data[kw_choice_id].values
        correct_choices = data[kw_correct_choice_id].values
        rts = data[kw_rt_id].values
        participant_id = data[kw_participant_id].values
        
        # get relevant meta data
        age = meta_data[meta_data[kw_participant_id]==participant_id[0]][kw_age].values + np.zeros_like(participant_id)
        gender = meta_data[meta_data[kw_participant_id]==participant_id[0]][kw_gender].values + np.zeros_like(participant_id)
        
        rewards = rewards[choices != -1]
        rts = rts[choices != -1]
        participant_id = participant_id[choices != -1]
        choices = choices[choices != -1]
        correct_choices = correct_choices[choices != -1]

        all_data.append(np.stack((participant_id, age, gender, choices, rewards, correct_choices, rts), axis=-1))
    except Exception as e:
        print(e)
        print(f"Could not process file {file} because of some error...")
        

data = np.concatenate(all_data)

# normalizing
data[:, 1] = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max()-data[:, 1].min())
data[:, 2] = (data[:, 2] - data[:, 2].min()) / (data[:, 2].max()-data[:, 2].min())

data = pd.DataFrame(data, columns=columns_out)
print(data)
data.to_csv('data/eckstein2022_age_gender.csv', index=False)