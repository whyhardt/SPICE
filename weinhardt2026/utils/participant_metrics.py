"""Participant-level data for the beta effects and the morphing analysis.

`study_data(study)` is the study's data CSV with one reward column; `REWARD_COLUMNS[study]` = (column, participant
column, block column or None). A participant's reward rate (`reward_rate`) is the column's mean over their trials,
or, with a block column, its sum per block averaged over blocks:
    dezfouli2019, ganesh2024a, eckstein2026: 'reward' per trial as recorded
    bustamante2023: 'reward_per_second' — reward per second, per round, averaged over rounds: every round is a fixed
                  window (round_duration, 240 s), so a round's total reward / its duration is the time-based reward
                  rate (time spent travelling and missed responses lower it); each row holds reward / duration
    bruckner2025: 'trial_reward' = the coin value r_t if the bag was caught, else 0
    weber2024:    'laser_caught' (a caught beam is the task's reward)
    braun2018:    'points_per_second' — points per second of response time, per block, averaged over blocks. A block
                  ends when the points reach BRAUN_TARGET_POINTS (the reconstructed totals of the gap-free blocks are
                  495-508; participants are assumed correct on every trial), so block points = target. Time = the
                  block's summed RTs over all its trials (excluded trials, missing rows, at the block's mean RT), which
                  leaves out the response-stimulus interval (200 / 1100 ms by block). The per-block rate is spread
                  over the block's rows, so summing per block recovers it.
Further participant-level criteria (constant per participant, in study_data):
    braun2018:    'rt_switch_cost' — mean RT on switch minus repeat trials (RTs 200-3000 ms); undefined (NaN) for a
                  participant who never switches
Participant order follows the data, as csv_to_dataset numbers the participants.
"""

import os

import numpy as np
import pandas as pd

from spice.utils.analysis.coefficient_betas import get_criterion

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

REWARD_COLUMNS = {
    'dezfouli2019': ('reward', 'participant', None),
    'ganesh2024a': ('reward', 'subjID', None),
    'eckstein2026': ('reward', 'participant', None),
    'bustamante2023': ('reward_per_second', 'subject_id', 'overall_round'),
    'braun2018': ('points_per_second', 'subject', 'block'),
    'bruckner2025': ('trial_reward', 'participant', None),
    'weber2024': ('laser_caught', 'participant', None),
}
BRAUN_TARGET_POINTS = 500


def study_data(study: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, 'weinhardt2026', 'studies', study, 'data', f'{study}.csv'), low_memory=False)
    if study == 'bruckner2025':
        df['trial_reward'] = df['r_t'] * df['catch']
    if study == 'braun2018':
        blocks = df.groupby(['subject', 'block'])
        seconds = blocks['rt'].transform('mean') / 1000 * blocks['trial'].transform('max')   # trials are numbered from 1
        df['points_per_second'] = BRAUN_TARGET_POINTS / seconds / blocks['trial'].transform('size')
        rt = df['rt'].where(df['rt'].between(200, 3000))
        switch_cost = rt[df['transcode'] == 1].groupby(df['subject']).mean() - rt[df['transcode'] == 0].groupby(df['subject']).mean()
        df['rt_switch_cost'] = df['subject'].map(switch_cost)
    if study == 'bustamante2023':
        df['reward_per_second'] = df['reward'] / (df['round_duration'] / 1000)
    return df


def reward_rate(study: str) -> np.ndarray:
    """Mean reward per trial of every participant, in model (data) order."""
    if study not in REWARD_COLUMNS:
        raise ValueError(f'{study}: no reward column (see weinhardt2026/utils/participant_metrics.py)')
    column, participant, block = REWARD_COLUMNS[study]
    return get_criterion(study_data(study), column, participant, block_col=block)[column].to_numpy()

