import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_behavioral_metrics(df, dataset_name):

    unique_sessions = df['session'].unique()
    print(f"Found {len(unique_sessions)} unique sessions")
    
    behavior_metrics = []
    
    for pid in tqdm(unique_sessions, desc=f"Processing {dataset_name}"):
        participant_df = df[df['session'] == pid]
        if participant_df.empty:
            continue
        
        choices = participant_df['choice'].values
        rewards = participant_df['reward'].values
        
        # Stay after reward rate
        stay_after_reward_count = 0
        stay_after_reward_total = 0
        for i in range(len(choices) - 1):
            current_choice = choices[i]
            next_choice = choices[i+1]
            current_reward = rewards[i]
            if current_reward > 0:
                stay_after_reward_total += 1
                if next_choice == current_choice:
                    stay_after_reward_count += 1
        stay_after_reward_rate = stay_after_reward_count / stay_after_reward_total if stay_after_reward_total > 0 else 0
        
        # Switch rate calculation
        switch_count = 0
        switch_total = 0
        for i in range(len(choices) - 1):
            current_choice = choices[i]
            next_choice = choices[i+1]
            switch_total += 1
            if next_choice != current_choice:
                switch_count += 1
        switch_rate = switch_count / switch_total if switch_total > 0 else 0
        
        # Perseveration: staying with same choice after 3+ consecutive unrewarded trials
        perseveration_count = 0
        perseveration_total = 0
        
        for i in range(3, len(choices)):  # Start from trial 4 (index 3) to check previous 3 trials
            # Check if previous 3 trials were unrewarded and same choice
            prev_3_rewards = rewards[i-3:i]
            prev_3_choices = choices[i-3:i]
            current_choice = choices[i]
            
            # If all previous 3 trials were unrewarded (0) and same choice
            if (np.all(prev_3_rewards == 0) and 
                np.all(prev_3_choices == prev_3_choices[0]) and 
                len(np.unique(prev_3_choices)) == 1):
                perseveration_total += 1
                # Check if current trial continues with same choice (perseveration)
                if current_choice == prev_3_choices[0]:
                    perseveration_count += 1
        
        perseveration = perseveration_count / perseveration_total if perseveration_total > 0 else 0
        
        # Stay behavior based on previous 2 outcomes
        stay_after_plus_plus = 0  # Stay after ++ (both previous rewarded)
        stay_after_plus_minus = 0  # Stay after +- 
        stay_after_minus_plus = 0  # Stay after -+
        stay_after_minus_minus = 0  # Stay after -- (both previous unrewarded)
        
        total_plus_plus = 0
        total_plus_minus = 0  
        total_minus_plus = 0
        total_minus_minus = 0
        
        for i in range(2, len(choices) - 1):  # Start from trial 3, need next trial to evaluate stay
            current_choice = choices[i]   # t
            next_choice = choices[i+1]    # t+1
            
            prev_reward_1 = rewards[i-2]  # reward at t-2
            prev_reward_2 = rewards[i-1]  # reward at t-1
            
            # Categorize based on previous 2 outcomes
            if prev_reward_1 > 0 and prev_reward_2 > 0:  # ++
                total_plus_plus += 1
                if next_choice == current_choice:
                    stay_after_plus_plus += 1
            elif prev_reward_1 > 0 and prev_reward_2 == 0:  # +-
                total_plus_minus += 1
                if next_choice == current_choice:
                    stay_after_plus_minus += 1
            elif prev_reward_1 == 0 and prev_reward_2 > 0:  # -+
                total_minus_plus += 1
                if next_choice == current_choice:
                    stay_after_minus_plus += 1
            elif prev_reward_1 == 0 and prev_reward_2 == 0:  # --
                total_minus_minus += 1
                if next_choice == current_choice:
                    stay_after_minus_minus += 1
        
        # Calculate stay rates for each condition
        stay_after_plus_plus_rate = stay_after_plus_plus / total_plus_plus if total_plus_plus > 0 else np.nan
        stay_after_plus_minus_rate = stay_after_plus_minus / total_plus_minus if total_plus_minus > 0 else np.nan
        stay_after_minus_plus_rate = stay_after_minus_plus / total_minus_plus if total_minus_plus > 0 else np.nan
        stay_after_minus_minus_rate = stay_after_minus_minus / total_minus_minus if total_minus_minus > 0 else np.nan
        
        # Average reward and reaction time
        avg_reward = participant_df['reward'].mean()
        avg_rt = participant_df['rt'].mean() if 'rt' in participant_df.columns else np.nan
        
        participant_data = {
            'participant_id': pid,
            'stay_after_reward': stay_after_reward_rate,
            'switch_rate': switch_rate,
            'perseveration': perseveration,
            'stay_after_plus_plus': stay_after_plus_plus_rate,
            'stay_after_plus_minus': stay_after_plus_minus_rate,
            'stay_after_minus_plus': stay_after_minus_plus_rate,
            'stay_after_minus_minus': stay_after_minus_minus_rate,
            'avg_reward': avg_reward,
            'avg_rt': avg_rt,
            'n_trials': len(participant_df),
            'dataset': dataset_name
        }
        
        behavior_metrics.append(participant_data)
    
    return pd.DataFrame(behavior_metrics)



#change to benchmark!!!!!!
file_paths = {
    'ApBr': "/Users/martynaplomecka/closedloop_rl/params/benchmarking/eckstein2022_simulated_ApBr.csv",
    'RNN': "/Users/martynaplomecka/closedloop_rl/params/benchmarking/eckstein2022_simulated_rnn_new.csv",
    'SPICE': "/Users/martynaplomecka/closedloop_rl/params/benchmarking/eckstein2022_simulated_spice_new.csv",
    'True': "/Users/martynaplomecka/closedloop_rl/params/benchmarking/eckstein2022_age.csv",
    'Benchmark': "/Users/martynaplomecka/closedloop_rl/params/benchmarking/eckstein2022_simulated_benchmark.csv"

}

all_behavioral_data = []

for dataset_name, file_path in file_paths.items():
    try:
        print(f"\nLoading {dataset_name} from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        behavioral_df = calculate_behavioral_metrics(df, dataset_name)
        all_behavioral_data.append(behavioral_df)
        
        output_filename = f"behavioral_metrics_{dataset_name.lower()}.csv"
        behavioral_df.to_csv(output_filename, index=False)
        print(f"Saved {len(behavioral_df)} participants to {output_filename}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")

if all_behavioral_data:
    combined_df = pd.concat(all_behavioral_data, ignore_index=True)
    combined_df.to_csv("new_behavioral_metrics_all_simulated.csv", index=False)

else:
    print("error.")