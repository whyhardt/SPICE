import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# Update this path to your local raw data directory
path = 'weinhardt2026/studies/archive/weber2024/data_raw'
target_file = 'weinhardt2026/studies/archive/weber2024/data/weber2024.csv'
data_condition = lambda filename: 'baseline' in filename

# Safely identify all participant directories
dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
all_subject_events = [] 

for d in tqdm(dirs):
    dir_path = os.path.join(path, d)
    
    # Gather and alphabetically sort files to guarantee chronological session tracking
    files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and data_condition(f)])
    
    # Cumulative block offset across files for this participant
    block_offset = 0
    for file_idx, f in enumerate(files):
        file_path = os.path.join(dir_path, f)
        df = pd.read_csv(file_path)
        
        # Extract participant ID safely using regex digits from the folder name
        match = re.search(r'\d+', d)
        df['participant'] = float(match.group()) if match else np.nan
        
        # Factorial Experiment ID mapping (0: 0x0, 1: 0x1, 2: 1x0, 3: 1x1)
        df['experiment'] = (df['volatility'].astype(int) * 2) + df['stochasticity'].astype(int)
        
        # --- FRAME-LEVEL PREPROCESSING ---
        
        # Segment continuous time-series into events based on when a new laser position appears
        df['is_new_laser'] = df['laserRotation'].diff() != 0
        df.loc[df.index[0], 'is_new_laser'] = True
        df['event_id'] = df['is_new_laser'].cumsum()
        
        # Frame-by-frame Stay/Move metrics (1 if shield moved, 0 if stationary)
        df['delta_shield'] = df['shieldRotation'].diff().fillna(0)
        df['is_moving'] = (df['delta_shield'] != 0).astype(int)
        
        # Movement Onset: Captures when they transition from staying (0) to moving (1)
        df['movement_onset'] = (df['is_moving'].diff() == 1).astype(int)
        if df['is_moving'].iloc[0] == 1:
            df.loc[df.index[0], 'movement_onset'] = 1
            
        # Circular Geometry Mapping to resolve shortest angular distances
        df['shield_mapped'] = df['shieldRotation'] % 360
        raw_diff = np.abs(df['shield_mapped'] - df['laserRotation'])
        df['shortest_distance'] = np.minimum(raw_diff, 360 - raw_diff)
        
        # Check if the laser fell within the shield width radius during this frame
        df['is_covering_laser'] = df['shortest_distance'] <= (df['shieldDegrees'] / 2)
        
        # --- AGGREGATING FRAMES INTO EVENTS ---
        event_rows = []
        prev_total_reward = df['totalReward'].iloc[0]  # initial cumulative reward before first event
        for event_id, group in df.groupby('event_id'):
            
            moving_frames = np.where(group['is_moving'] == 1)[0]
            
            # Reaction Time: total frames elapsed until the first active movement frame
            reaction_time = moving_frames[0] if len(moving_frames) > 0 else len(group)
            
            # Compute continuous, 0-indexed block ID across multiple chronological files per participant
            raw_block = group['blockID'].iloc[0]
            global_block_id = (raw_block - 1) + block_offset
            
            event_rows.append({
                # Metadata
                'participant': group['participant'].iloc[0],
                'experiment': group['experiment'].iloc[0],
                'block': global_block_id, # Chained continuously between [0, n_files * 4 - 1]
                'event': event_id,
                
                # Environmental/Task Variables
                'volatility': group['volatility'].iloc[0],
                'stochasticity': group['stochasticity'].iloc[0],
                'laserRotation': group['laserRotation'].iloc[0],
                'shieldRotation': group['shieldRotation'].iloc[0],
                
                # TARGET VARIABLE (Stay/Move Space)
                # 1 if they moved at all during this event window, 0 if they stayed completely still
                'action': 1 if group['is_moving'].any() else 0,
                
                # PI Metric: trialDuration (until next event)
                'trial_duration_frames': len(group),
                
                # PI Metric: shieldDistance (Shortest circular angular distance at trial onset)
                'shield_distance_initial': group['shortest_distance'].iloc[0],
                
                # PI Metrics: totalMovement & Execution Tracking
                'total_movement_degrees': group['delta_shield'].abs().sum(),
                'frames_spent_moving': group['is_moving'].sum(),
                'button_press_onsets': group['movement_onset'].sum(),
                'reaction_time_frames': reaction_time,
                
                # PI Metric: reward assessment metrics
                'laser_caught': group['is_covering_laser'].any(),
                'hit_occurred': group['currentHit'].any(),
                'reward_change': group['totalReward'].iloc[0] - prev_total_reward,
                'total_reward': group['totalReward'].iloc[0],
            })
            prev_total_reward = group['totalReward'].iloc[-1]
            
        block_offset += df['blockID'].nunique()
        all_subject_events.append(pd.DataFrame(event_rows))

# --- MASTER POST-PROCESSING & PREDICTIVE FEATURE SHIFT ---
if all_subject_events:
    df_master = pd.concat(all_subject_events, ignore_index=True)
    
    # 1. Sort dataset chronologically across the new global block chain
    df_master = df_master.sort_values(by=['participant', 'block', 'event']).reset_index(drop=True)
    
    # 2. Add the look-ahead feature for next action/stimulus modeling
    # Bounding this shift by participant and global block prevents cross-block data leakage
    df_master['next_laserRotation'] = df_master.groupby(['participant', 'block'])['laserRotation'].shift(-1)
    
    # Reordering columns strictly to place the target variable and lookahead feature cleanly together
    columns_ordered = [
        'participant', 'experiment', 'block', 'event', 
        'volatility', 'stochasticity', 'shieldRotation', 'laserRotation', 'next_laserRotation', 'action',
        'trial_duration_frames', 'shield_distance_initial', 'total_movement_degrees', 
        'frames_spent_moving', 'button_press_onsets', 'reaction_time_frames', 
        'laser_caught', 'hit_occurred', 'reward_change', 'total_reward'
    ]
    df_master = df_master[columns_ordered]
    
    # Export final master aggregated event file 
    df_master.to_csv(target_file, index=False)
    print(f"Pipeline executed successfully! Master file generated with {len(df_master)} total unique events.")
else:
    print("Error: No matching data files found. Check your directory path or data_condition.")