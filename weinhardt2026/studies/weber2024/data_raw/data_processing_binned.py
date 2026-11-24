"""Frame-binned weber2024 dataset: every block's raw frames are cut into fixed bins of BIN_FRAMES frames.

One row per bin. A beam is presented on a bin only if a new laser position appears within it
(`beam_presented` = 1); on every other bin the laser carries no new information (`beam_presented` = 0,
laser columns hold the last beam for reference only). Longer gaps between beams are therefore more bins
without a beam, not a larger input value.

With BIN_FRAMES = 5 every beam keeps its own bin: beams are >= 6 frames apart, except for the 1-4 frame
segments the task cuts off when a changepoint fires a new beam immediately. When two onsets share a bin, the
bin keeps the later one (the beam after the changepoint) and `n_beams_in_bin` records the collision.
"""

import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

BIN_FRAMES = 5

path = 'weinhardt2026/studies/weber2024/data_raw'
target_file = f'weinhardt2026/studies/weber2024/data/weber2024_bin{BIN_FRAMES}.csv'
data_condition = lambda filename: 'baseline' in filename


def circular_distance(a, b):
    diff = np.abs((a % 360) - (b % 360))
    return np.minimum(diff, 360 - diff)


def bin_block(block: pd.DataFrame) -> pd.DataFrame:
    """Bin one block's frames; returns one row per bin."""
    block = block.reset_index(drop=True)
    n_frames = len(block)
    laser = block['laserRotation'].to_numpy()
    shield = block['shieldRotation'].to_numpy()

    onset = np.r_[True, np.diff(laser) != 0]
    covering = circular_distance(shield, laser) <= block['shieldDegrees'].to_numpy() / 2
    moving = np.r_[False, np.diff(shield) != 0]

    bin_index = np.arange(n_frames) // BIN_FRAMES
    rows = []
    for b in range(bin_index[-1] + 1):
        frames = np.flatnonzero(bin_index == b)
        onsets = frames[onset[frames]]
        # the bin's beam: the last onset within it (if any); its onset frame scores the catch and the reward
        beam_frame = onsets[-1] if len(onsets) else None
        last_onset = np.flatnonzero(onset[:frames[-1] + 1])[-1]   # most recent beam up to the bin's end
        rows.append({
            'bin': b,
            'bin_frames': len(frames),
            'beam_presented': int(beam_frame is not None),
            'n_beams_in_bin': len(onsets),
            'onset_frame_in_bin': int(beam_frame - frames[0]) if beam_frame is not None else np.nan,
            'laserRotation': laser[last_onset],
            'shieldRotation': shield[frames[0]],                   # shield at the bin's start
            'shieldRotation_end': shield[frames[-1]],              # shield at the bin's end
            'action': int(moving[frames].any()),                   # moved within the bin
            'laser_caught': bool(covering[beam_frame]) if beam_frame is not None else False,
            'hit_occurred': bool(block['currentHit'].to_numpy()[frames].any()),
            'total_reward': block['totalReward'].to_numpy()[frames[-1]],
            'trueMean': block['trueMean'].to_numpy()[frames[-1]],
        })
    binned = pd.DataFrame(rows)
    binned['reward_change'] = binned['total_reward'].diff().fillna(
        binned['total_reward'].iloc[0] - block['totalReward'].iloc[0])
    return binned


def main():
    dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    blocks = []
    for d in tqdm(dirs):
        participant = float(re.search(r'\d+', d).group())
        files = sorted(f for f in os.listdir(os.path.join(path, d)) if data_condition(f))
        block_offset = 0
        for f in files:
            df = pd.read_csv(os.path.join(path, d, f))
            for raw_block, block in df.groupby('blockID'):
                binned = bin_block(block)
                binned.insert(0, 'participant', participant)
                binned.insert(1, 'experiment', int(block['volatility'].iloc[0]) * 2 + int(block['stochasticity'].iloc[0]))
                binned.insert(2, 'block', (raw_block - 1) + block_offset)
                binned.insert(4, 'volatility', block['volatility'].iloc[0])
                binned.insert(5, 'stochasticity', block['stochasticity'].iloc[0])
                blocks.append(binned)
            block_offset += df['blockID'].nunique()

    data = pd.concat(blocks, ignore_index=True).sort_values(['participant', 'block', 'bin']).reset_index(drop=True)
    data['next_shieldRotation'] = data.groupby(['participant', 'block'])['shieldRotation'].shift(-1)
    data.to_csv(target_file, index=False)
    print(f'{len(data)} bins, {int(data.beam_presented.sum())} bins with a beam, '
          f'{int((data.n_beams_in_bin > 1).sum())} bins with >1 onset -> {target_file}')


if __name__ == '__main__':
    main()
