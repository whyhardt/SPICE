import numpy as np
import pandas as pd
import torch

from spice import SpiceDataset, csv_to_dataset, split_data_along_blockdim

from weinhardt2026.studies.kolff2025.spice_kolff2025_groom_id2 import CONFIG, prepare_dataset, target_mask


# written by preprocessing_kolff2025.py
DEFAULT_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_categories.csv'
ID2_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_groom_id2.csv'
MIRRORED_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_groom_mirrored.csv'

# ID1's acts as ID2 receives them, grouped as in the TOBETO split (data/PREPROCESSING.md):
# a to-be-groomed signal from ID1 is negotiation, a to-groom signal is preparation for
# grooming ID2, and Groom is grooming ID2.
ID1_ACT_GROUPS = {
    'id1_negotiation': ('Grooming_Solicitation', 'Self_Reposition', 'Directed_Scratch'),
    'id1_prepare': ('Reposition_Body', 'Grooming_Process'),
    'id1_groom': ('Groom',),
}
# ID2's own acts at the current event, same grouping
ID2_ACT_GROUPS = {
    'id2_negotiation': ('Grooming_Solicitation', 'Self_Reposition', 'Directed_Scratch'),
    'id2_prepare': ('Reposition_Body', 'Grooming_Process'),
    'id2_groom': ('Groom',),
}
# single categories (data/PREPROCESSING.md) as indicator columns id1_<category> / id2_<category>
CATEGORIES = ('Grooming_Solicitation', 'Self_Reposition', 'Directed_Scratch', 'Reposition_Body', 'Grooming_Process', 'Groom')
# single-category input columns (spice_kolff2025_groom_id2_categories / _shared)
FINE_ID1_COLUMNS = tuple(f'id1_{category.lower()}' for category in CATEGORIES)
FINE_ID2_COLUMNS = tuple(f'id2_{category.lower()}' for category in CATEGORIES)
# last event for the lookup benchmark: ID2's act group if ID2 acted, else ID1's
LAST_EVENTS = ('none',) + tuple(ID1_ACT_GROUPS) + tuple(ID2_ACT_GROUPS)


# -------------------------------------------------------------------------------------------
# ID2 PERSPECTIVE
# -------------------------------------------------------------------------------------------

def build_id2_dataframe(path_data: str = None) -> pd.DataFrame:
    """One sequence per interaction, from the approached ape's (ID2's) point of view.

    ID1 (the approacher) and ID2 are taken from the interaction's first row; in two
    interactions some rows list the apes the other way round, so acts are assigned by ape
    name. Every event is kept as an input step; the target at event t is whether ID2 grooms
    at event t+1, defined only if ID2 acts at t+1 (`id2_acts_next`).

    Dominance ranks are normalized within community (hierarchies of depth 27 and 15).
    `rank_diff_centered` is ID2's rank minus ID1's, centred within ID2 so that it is
    orthogonal to the per-ape bias.
    """
    df = pd.read_csv(path_data if path_data is not None else DEFAULT_DATA_PATH)

    ranks_long = pd.concat([
        df[['community_id', 'rank_ID1']].rename(columns={'rank_ID1': 'rank'}),
        df[['community_id', 'rank_ID2']].rename(columns={'rank_ID2': 'rank'}),
    ])
    max_rank = ranks_long.groupby('community_id')['rank'].max().to_dict()

    records = []
    for interaction_id, bout in df.groupby('interaction_id', sort=False):
        id1, id2 = bout['ID1'].iloc[0], bout['ID2'].iloc[0]
        for event, row in enumerate(bout.itertuples()):
            act_id1 = row.Category_ID1 if row.ID1 == id1 else row.Category_ID2
            act_id2 = row.Category_ID2 if row.ID2 == id2 else row.Category_ID1
            rank_id1 = row.rank_ID1 if row.ID1 == id1 else row.rank_ID2
            rank_id2 = row.rank_ID2 if row.ID2 == id2 else row.rank_ID1
            scale = max_rank[row.community_id]
            records.append({
                'participant': id2,
                'partner': id1,
                'interaction_id': interaction_id,
                'event': event,
                **{group: float(act_id1 in categories) for group, categories in ID1_ACT_GROUPS.items()},
                **{group: float(act_id2 in categories) for group, categories in ID2_ACT_GROUPS.items()},
                **{f'id1_{category.lower()}': float(act_id1 == category) for category in CATEGORIES},
                **{f'id2_{category.lower()}': float(act_id2 == category) for category in CATEGORIES},
                'id2_acts': float(isinstance(act_id2, str)),
                'rank_own': rank_id2 / scale,
                'rank_partner': rank_id1 / scale,
                'community_id': row.community_id,
            })

    out = pd.DataFrame.from_records(records)
    out['rank_diff'] = out['rank_own'] - out['rank_partner']
    out['rank_diff_centered'] = out['rank_diff'] - out.groupby('participant')['rank_diff'].transform('mean')
    # scaled to [-1, 1] like the other inputs (raw range about -0.8 .. 0.5)
    out['rank_diff_centered'] = out['rank_diff_centered'] / out['rank_diff_centered'].abs().max()
    # own rank, which a group model cannot absorb into an embedding, scaled from [0, 1] to [-1, 1]
    out['rank_own_scaled'] = 2 * out['rank_own'] - 1
    # readout: ID2 grooms (1) or not (0); csv_to_dataset shifts it by one event
    out['choice'] = out['id2_groom']
    # which targets exist: does ID2 act at the next event?
    out['id2_acts_next'] = out.groupby('interaction_id')['id2_acts'].shift(-1).fillna(0.)
    return out


def build_mirrored_dataframe(path_data: str = None) -> pd.DataFrame:
    """Every interaction twice: from the approached ape's (ID2's) and the approacher's (ID1's)
    point of view. Used by spice_kolff2025_groom_mirrored.

    Columns use self_* (the participant's own act) and partner_* (the other ape's act) per
    category, and `is_approacher` flags the ID1 perspective. The target at event t is whether
    the participant grooms at event t+1, defined only if it acts at t+1 (`self_acts_next`).
    Both perspectives of an interaction share its interaction_id, so a block split keeps them
    on the same side. `rank_diff_centered` (own minus partner rank) is centred within the
    participant over both roles and scaled to [-1, 1].
    """
    df = pd.read_csv(path_data if path_data is not None else DEFAULT_DATA_PATH)

    ranks_long = pd.concat([
        df[['community_id', 'rank_ID1']].rename(columns={'rank_ID1': 'rank'}),
        df[['community_id', 'rank_ID2']].rename(columns={'rank_ID2': 'rank'}),
    ])
    max_rank = ranks_long.groupby('community_id')['rank'].max().to_dict()

    records = []
    for interaction_id, bout in df.groupby('interaction_id', sort=False):
        id1, id2 = bout['ID1'].iloc[0], bout['ID2'].iloc[0]
        for focal, other in ((id2, id1), (id1, id2)):
            for event, row in enumerate(bout.itertuples()):
                act_self = row.Category_ID1 if row.ID1 == focal else row.Category_ID2
                act_partner = row.Category_ID1 if row.ID1 == other else row.Category_ID2
                rank_self = row.rank_ID1 if row.ID1 == focal else row.rank_ID2
                rank_partner = row.rank_ID1 if row.ID1 == other else row.rank_ID2
                scale = max_rank[row.community_id]
                records.append({
                    'participant': focal,
                    'partner': other,
                    'interaction_id': interaction_id,
                    'event': event,
                    'is_approacher': float(focal == id1),
                    **{f'partner_{category.lower()}': float(act_partner == category) for category in CATEGORIES},
                    **{f'self_{category.lower()}': float(act_self == category) for category in CATEGORIES},
                    'self_acts': float(isinstance(act_self, str)),
                    'rank_own': rank_self / scale,
                    'rank_partner': rank_partner / scale,
                    'community_id': row.community_id,
                })

    out = pd.DataFrame.from_records(records)
    out['rank_diff'] = out['rank_own'] - out['rank_partner']
    out['rank_diff_centered'] = out['rank_diff'] - out.groupby('participant')['rank_diff'].transform('mean')
    out['rank_diff_centered'] = out['rank_diff_centered'] / out['rank_diff_centered'].abs().max()
    # own rank, which a group model cannot absorb into an embedding, scaled from [0, 1] to [-1, 1]
    out['rank_own_scaled'] = 2 * out['rank_own'] - 1
    # readout: the participant grooms (1) or not (0); csv_to_dataset shifts it by one event
    out['choice'] = out['self_groom']
    out['self_acts_next'] = out.groupby(['participant', 'interaction_id'])['self_acts'].shift(-1).fillna(0.)
    return out


def drop_test_only_participants(df: pd.DataFrame, test_blocks: list[int]) -> pd.DataFrame:
    """Remove the sessions of participants that occur only in test interactions.

    Such an ape has no training data (its embedding would stay untrained), and run.py sizes the
    participant embedding by the apes in the training split, so its id would fall out of range.
    In the mirrored data this only removes the ape's own perspective; the interactions stay in
    the data from the partner's perspective.
    """
    train_apes = set(df.loc[~df['interaction_id'].isin(test_blocks), 'participant'])
    dropped = sorted(set(df['participant']) - train_apes)
    if dropped:
        print(f'drop_test_only_participants: removing {dropped}')
    return df[df['participant'].isin(train_apes)].reset_index(drop=True)


# -------------------------------------------------------------------------------------------
# BLOCK SPLIT WITHIN KNOWN DYADS
# -------------------------------------------------------------------------------------------

def _draw_test_interactions(bouts: pd.DataFrame, eligible: list, test_fraction: float, rng: np.random.Generator) -> set:
    """One random test set: at most one interaction per eligible dyad, until about `test_fraction` of the targets."""
    total, held, test = bouts['n_targets'].sum(), 0, set()
    for index in rng.permutation(len(eligible)):
        if held >= test_fraction * total:
            break
        candidates = bouts.index[bouts['dyad'] == eligible[index]]
        interaction = candidates[rng.integers(len(candidates))]
        test.add(interaction)
        held += bouts.loc[interaction, 'n_targets']
    return test


def select_test_interactions(df: pd.DataFrame, test_fraction: float, seed: int, n_draws: int = 2000) -> list[int]:
    """Hold out single interactions of known dyads (block split), balanced for the ID2 grooming ratio.

    As in the other studies, the test set holds new sessions (bouts) of known participants: only
    dyads with at least two interactions are eligible, and each contributes at most one test
    interaction, so every test dyad and every test ape stays in the training data. Whole new
    dyads are not tested -- a new partner is a new condition.

    Only interactions with at least one target count. `n_draws` random test sets are drawn and
    the one closest to the training set is kept: same P(ID2 grooms) over targets, same share of
    targets in interactions where ID2 never grooms, and a test share of targets near
    `test_fraction`. Uses the data only (no model), and is deterministic given `seed`.
    """
    targets = df[df['id2_acts_next'] == 1].assign(
        y=df.groupby('interaction_id')['id2_groom'].shift(-1)[df['id2_acts_next'] == 1])
    bouts = targets.groupby('interaction_id').agg(participant=('participant', 'first'), partner=('partner', 'first'),
                                                  n_targets=('y', 'size'), n_groom=('y', 'sum'))
    bouts['dyad'] = [tuple(sorted(pair)) for pair in zip(bouts['participant'], bouts['partner'])]
    n_per_dyad = bouts.groupby('dyad').size()
    eligible = sorted(n_per_dyad[n_per_dyad >= 2].index)

    def stats(b):
        return b['n_groom'].sum() / b['n_targets'].sum(), b.loc[b['n_groom'] == 0, 'n_targets'].sum() / b['n_targets'].sum()

    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(n_draws):
        is_test = bouts.index.isin(list(_draw_test_interactions(bouts, eligible, test_fraction, rng)))
        (p_test, never_test), (p_train, never_train) = stats(bouts[is_test]), stats(bouts[~is_test])
        share = bouts.loc[is_test, 'n_targets'].sum() / bouts['n_targets'].sum()
        score = abs(p_test - p_train) + 0.5 * abs(never_test - never_train) + 0.5 * abs(share - test_fraction)
        if score < best_score:
            best, best_score = is_test, score

    return [int(i) for i in bouts.index[best]]


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

DATA_KWARGS = dict(
    df_participant_id='participant',
    df_block='interaction_id',
    df_choice='choice',
    df_feedback=None,
    additional_inputs=CONFIG.additional_inputs,
)


def get_dataset(
    path_data: str = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    save_csv: str = ID2_DATA_PATH,
) -> tuple[SpiceDataset, SpiceDataset, dict]:
    """Build the ID2 grooming dataset: ys = [not groom, groom, target mask] (see prepare_dataset)."""
    df = build_id2_dataframe(path_data)
    if save_csv:
        df.to_csv(save_csv, index=False)

    test_blocks = select_test_interactions(df, test_fraction, seed) if test_fraction > 0 else []
    dataset = prepare_dataset(csv_to_dataset(file=df, **DATA_KWARGS))
    if test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train, dataset_test = dataset, dataset

    info = {
        'n_participants': dataset.n_participants,
        'n_sessions_train': dataset_train.xs.shape[0],
        'n_sessions_test': dataset_test.xs.shape[0],
        'n_targets_train': int(target_mask(dataset_train.ys).sum()),
        'n_targets_test': int(target_mask(dataset_test.ys).sum()),
        'p_groom': float(df.loc[df['id2_acts'] == 1, 'id2_groom'].mean()),
        'test_blocks': test_blocks,
    }
    return dataset_train, dataset_test, info


def evaluation_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """ys reduced to [not groom, groom] and NaN where ID2 does not act next.

    analysis_model_evaluation then scores only real targets when called with
    trial_filter=has_target.
    """
    ys = dataset.ys[..., :2].clone()
    ys[~target_mask(dataset.ys)] = float('nan')
    return SpiceDataset(dataset.xs, ys, n_reward_features=0)


def has_target(ys: torch.Tensor) -> torch.Tensor:
    """trial_filter for analysis_model_evaluation on an evaluation_dataset."""
    return ~torch.isnan(ys[:, :, 0, 0])


# -------------------------------------------------------------------------------------------
# GRU WITH THE SPICE MODEL'S INPUTS
# -------------------------------------------------------------------------------------------

from weinhardt2026.utils.benchmarking_gru import GRUModel


class ModelInputGRU(GRUModel):
    """GRU (no participant embedding) that sees what the SPICE model sees.

    `additional_inputs` are the model's input columns in data order (default: the broad act
    groups of spice_kolff2025_groom_id2; pass a model's CONFIG.additional_inputs for its
    inputs, e.g. the single categories). The last one is the target mask (future information)
    and is excluded; the choice input (a duplicate of ID2's own grooming) is zeroed."""

    def __init__(self, hidden_size: int = 8, additional_inputs: tuple = None, **kwargs):
        additional_inputs = CONFIG.additional_inputs if additional_inputs is None else additional_inputs
        super().__init__(n_actions=2, additional_inputs=len(additional_inputs) - 1,
                         n_reward_features=0, hidden_size=hidden_size, **kwargs)

    def forward(self, inputs, state=None):
        inputs = inputs.clone()
        inputs[..., :self.n_actions] = 0.
        return super().forward(inputs, state)


# -------------------------------------------------------------------------------------------
# BENCHMARK: LAG-1 LOOKUP TABLE ON THE LAST EVENT
# -------------------------------------------------------------------------------------------

class ConditionalGroomModel(torch.nn.Module):
    """Baseline: P(ID2 grooms next | last event), per ID2 ape.

    The last event is ID2's act if ID2 acted, else ID1's: the same inputs as the SPICE model
    but no memory. By default the acts are the broad groups (`LAST_EVENTS`, 7 cells); pass the
    model's `additional_inputs` with `id1_columns` / `id2_columns` (e.g. FINE_ID1_COLUMNS /
    FINE_ID2_COLUMNS) for the single categories (13 cells). Per-ape cells are shrunk toward the
    pooled table (prior strength `shrinkage` pseudo-observations), so apes with few targets fall
    back to the group instead of a uniform prior.
    """

    def __init__(self, n_participants: int, shrinkage: float = 5.0, additional_inputs: tuple = None,
                 id1_columns: tuple = None, id2_columns: tuple = None):
        super().__init__()
        self.n_participants = n_participants
        self.shrinkage = shrinkage
        self.additional_inputs = tuple(CONFIG.additional_inputs if additional_inputs is None else additional_inputs)
        self.id1_columns = tuple(ID1_ACT_GROUPS if id1_columns is None else id1_columns)
        self.id2_columns = tuple(ID2_ACT_GROUPS if id2_columns is None else id2_columns)
        self.n_events = 1 + len(self.id1_columns) + len(self.id2_columns)
        self.register_buffer('log_probs', torch.zeros(n_participants, self.n_events, 2))

    def _unpack(self, xs: torch.Tensor):
        start = 2  # [choice one-hot (2), additional inputs ..., 5 metadata]
        def act_index(columns):
            cols = [start + self.additional_inputs.index(c) for c in columns]
            onehot = xs[:, :, 0, cols].nan_to_num(0.)
            return torch.where(onehot.sum(-1) > 0, onehot.argmax(-1) + 1, torch.zeros_like(onehot[..., 0])).long()
        act_id1, act_id2 = act_index(self.id1_columns), act_index(self.id2_columns)
        event = torch.where(act_id2 > 0, act_id2 + len(self.id1_columns), act_id1)
        return xs[:, 0, 0, -1].nan_to_num(0).long(), event, ~torch.isnan(xs[:, :, 0, 0])

    def fit(self, dataset: SpiceDataset) -> 'ConditionalGroomModel':
        participant_ids, act, valid = self._unpack(dataset.xs.cpu())
        mask = valid & target_mask(dataset.ys.cpu())[:, :, 0]
        target = dataset.ys.cpu()[:, :, 0, :2].nan_to_num(0).argmax(-1)
        counts = torch.zeros(self.n_participants, self.n_events, 2)
        sessions, trials = torch.nonzero(mask, as_tuple=True)
        for s, t in zip(sessions.tolist(), trials.tolist()):
            counts[participant_ids[s], act[s, t], target[s, t]] += 1
        pooled = (counts.sum(0) + 1.) / (counts.sum(0) + 1.).sum(-1, keepdim=True)
        probs = (counts + self.shrinkage * pooled) / (counts.sum(-1, keepdim=True) + self.shrinkage)
        self.log_probs = torch.log(probs)
        return self

    def forward(self, xs: torch.Tensor, prev_state=None):
        participant_ids, act, valid = self._unpack(xs)
        logits = self.log_probs[participant_ids.unsqueeze(1).expand(-1, xs.shape[1]), act]  # (B, T, 2)
        logits[~valid] = float('nan')
        return logits.unsqueeze(2), None

    def count_parameters(self) -> int:
        # one free probability per last-event cell and ape
        return self.n_events
