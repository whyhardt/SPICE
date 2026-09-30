import numpy as np
import pandas as pd
import torch

from spice import SpiceDataset, csv_to_dataset, split_data_along_blockdim

from weinhardt2026.studies.kolff2025.spice_kolff2025_tobeto import (
    CONFIG, N_ITEMS, ITEM_NONE, ITEM_TOBEGROOMED, ITEM_TOGROOM, ITEM_GROOM, ITEM_NAMES,
)


# written by preprocessing_kolff2025.py
DEFAULT_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_categories.csv'
PERSPECTIVE_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_tobeto.csv'

ITEM_OF_CATEGORY = {
    'Grooming_Solicitation': ITEM_TOBEGROOMED,
    'Self_Reposition': ITEM_TOBEGROOMED,
    'Directed_Scratch': ITEM_TOBEGROOMED,
    'Reposition_Body': ITEM_TOGROOM,
    'Grooming_Process': ITEM_TOGROOM,
    'Groom': ITEM_GROOM,
}


def category_to_item(category) -> int:
    """Category -> TOBETO readout item; NaN (no act) -> ITEM_NONE."""
    return ITEM_NONE if pd.isna(category) else ITEM_OF_CATEGORY[category]


# -------------------------------------------------------------------------------------------
# PERSPECTIVE RESHAPING
# -------------------------------------------------------------------------------------------

def build_perspective_dataframe(path_data: str = None) -> pd.DataFrame:
    """Re-encode the dyadic event stream into one sequence per (focal ape, interaction).

    Every interaction is written out twice, once from each ape's point of view, so that a
    focal ape sees the complete stream of its own acts and the acts it received. ID1/ID2
    roles can swap within an interaction, so acts are assigned by ape name, not by column.

    Row order within an interaction is event order and is kept as `event`. Timestamps are
    not used: they restart with every video file, so sorting by them would scramble bouts.

    Dominance ranks are normalized *within community*: the two hierarchies differ in depth
    (1-27 and 1-15), so a one-step rank gap does not mean the same thing in both.

    Returns one row per (focal ape, event), with columns:
        focal, partner, interaction_id, event, own_act, partner_act,
        rank_own, rank_partner, rank_diff, community_id
    """

    df = pd.read_csv(path_data if path_data is not None else DEFAULT_DATA_PATH)

    ranks_long = pd.concat([
        df[['community_id', 'rank_ID1']].rename(columns={'rank_ID1': 'rank'}),
        df[['community_id', 'rank_ID2']].rename(columns={'rank_ID2': 'rank'}),
    ])
    max_rank = ranks_long.groupby('community_id')['rank'].max().to_dict()

    records = []
    for interaction_id, bout in df.groupby('interaction_id', sort=False):
        apes = sorted(set(bout['ID1']) | set(bout['ID2']))
        if len(apes) != 2:
            continue  # skip anything that is not strictly dyadic
        for focal in apes:
            partner = apes[1] if apes[0] == focal else apes[0]
            for event, row in enumerate(bout.itertuples()):
                if row.ID1 == focal:
                    own, other, rank_own, rank_partner = row.Category_ID1, row.Category_ID2, row.rank_ID1, row.rank_ID2
                else:
                    own, other, rank_own, rank_partner = row.Category_ID2, row.Category_ID1, row.rank_ID2, row.rank_ID1
                scale = max_rank[row.community_id]
                records.append({
                    'focal': focal,
                    'partner': partner,
                    'interaction_id': interaction_id,
                    'event': event,
                    'own_act': category_to_item(own),
                    'partner_act': category_to_item(other),
                    'rank_own': rank_own / scale,
                    'rank_partner': rank_partner / scale,
                    'community_id': row.community_id,
                })

    df_perspective = pd.DataFrame.from_records(records)
    df_perspective['rank_diff'] = df_perspective['rank_own'] - df_perspective['rank_partner']
    return df_perspective.sort_values(
        ['focal', 'interaction_id', 'event'], kind='stable',
    ).reset_index(drop=True)


# -------------------------------------------------------------------------------------------
# DYAD SPLIT
# -------------------------------------------------------------------------------------------

def select_test_interactions(df: pd.DataFrame, test_fraction: float, seed: int) -> list[int]:
    """Hold out whole dyads, keeping at least one training dyad for every ape.

    A held-out dyad is a partner pairing the model has never been fit on, so the test asks
    whether an ape's mechanism carries over to a new partner.
    """
    dyad_of_interaction = df.groupby('interaction_id').apply(
        lambda bout: tuple(sorted((bout['focal'].iloc[0], bout['partner'].iloc[0]))),
        include_groups=False,
    )
    dyads = sorted(set(dyad_of_interaction))

    n_train_dyads = {}
    for dyad in dyads:
        for ape in dyad:
            n_train_dyads[ape] = n_train_dyads.get(ape, 0) + 1

    rng = np.random.default_rng(seed)
    n_test = int(round(test_fraction * len(dyads)))
    test_dyads = set()
    for index in rng.permutation(len(dyads)):
        if len(test_dyads) >= n_test:
            break
        dyad = dyads[index]
        if all(n_train_dyads[ape] > 1 for ape in dyad):
            test_dyads.add(dyad)
            for ape in dyad:
                n_train_dyads[ape] -= 1

    return [int(i) for i, dyad in dyad_of_interaction.items() if dyad in test_dyads]


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

def get_dataset(
    path_data: str = None,
    test_fraction: float = 0.2,
    min_length: int = 2,
    seed: int = 42,
    save_csv: str = PERSPECTIVE_DATA_PATH,
) -> tuple[SpiceDataset, SpiceDataset, dict]:
    """Build the TOBETO dataset.

    One session = one (focal ape, interaction) perspective. The target at trial t is the
    focal ape's own act at t+1 over [none, tobegroomed, togroom, groom]. Held-out data are
    whole dyads (see select_test_interactions); both perspectives of a bout stay on the same
    side of the split.
    """

    df = build_perspective_dataframe(path_data)

    # a single event yields no target
    lengths = df.groupby(['focal', 'interaction_id'])['event'].transform('size')
    df = df[lengths >= min_length].copy()

    # Within-ape centring of the dominance distance: "is this partner more or less dominant
    # than my typical partner". Raw rank_diff is ~collinear with the ape's own rank
    # (rho = 0.92), and with per-ape coefficients its between-ape part is indistinguishable
    # from the bias term. Centred over the events actually fitted, so the orthogonality to
    # every per-ape variable is exact. Apes with a single partner get 0 throughout.
    df['rank_diff_centered'] = (
        df['rank_diff'] - df.groupby('focal')['rank_diff'].transform('mean')
    )

    # the readout target is the focal ape's own act (csv_to_dataset shifts it by one trial)
    df['choice'] = df['own_act']

    if save_csv:
        df.to_csv(save_csv, index=False)

    test_blocks = select_test_interactions(df, test_fraction, seed) if test_fraction > 0 else []

    dataset = csv_to_dataset(
        file=df,
        df_participant_id='focal',
        df_block='interaction_id',
        df_choice='choice',
        df_feedback=None,
        additional_inputs=CONFIG.additional_inputs,
    )

    if test_fraction > 0:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train, dataset_test = dataset, dataset

    info_dataset = {
        'n_participants': dataset.n_participants,
        'n_actions': dataset.n_actions,
        'n_sessions': dataset.xs.shape[0],
        'n_sessions_train': dataset_train.xs.shape[0],
        'n_sessions_test': dataset_test.xs.shape[0],
        'n_test_interactions': len(test_blocks),
        'item_frequencies': {
            ITEM_NAMES[k]: round(float(v), 4)
            for k, v in df['choice'].value_counts(normalize=True).sort_index().items()
        },
    }

    return dataset_train, dataset_test, info_dataset


# -------------------------------------------------------------------------------------------
# BENCHMARK: LAG-1 CONDITIONAL FREQUENCY MODEL
# -------------------------------------------------------------------------------------------

class ConditionalTobetoModel(torch.nn.Module):
    """Baseline: P(own act at t+1 | own act at t, partner act at t), per focal ape.

    A memoryless lookup table: no latent drive, no carry-over across events. Whatever SPICE
    gains over this is attributable to the latent dynamics.
    """

    def __init__(self, n_participants: int, smoothing: float = 1.0):
        super().__init__()
        self.n_participants = n_participants
        self.smoothing = smoothing
        self.register_buffer(
            'log_probs',
            torch.zeros(n_participants, N_ITEMS, N_ITEMS, N_ITEMS),
        )

    @staticmethod
    def _unpack(xs: torch.Tensor):
        # feature layout: [choice one-hot (N_ITEMS), own_act, partner_act, ..., 5 metadata cols]
        participant_ids = xs[:, 0, 0, -1].nan_to_num(0).long()
        own_acts = xs[:, :, 0, N_ITEMS].nan_to_num(ITEM_NONE).long().clamp(0, N_ITEMS - 1)
        partner_acts = xs[:, :, 0, N_ITEMS + 1].nan_to_num(ITEM_NONE).long().clamp(0, N_ITEMS - 1)
        valid = ~torch.isnan(xs[:, :, 0, 0])
        return participant_ids, own_acts, partner_acts, valid

    def fit(self, dataset: SpiceDataset) -> 'ConditionalTobetoModel':
        xs, ys = dataset.xs.cpu(), dataset.ys.cpu()
        participant_ids, own_acts, partner_acts, valid = self._unpack(xs)
        valid = valid & ~torch.isnan(ys[:, :, 0, 0])
        targets = ys[:, :, 0, :].nan_to_num(0).argmax(dim=-1)

        counts = torch.full((self.n_participants, N_ITEMS, N_ITEMS, N_ITEMS), self.smoothing)
        sessions, trials = torch.nonzero(valid, as_tuple=True)
        for s, t in zip(sessions.tolist(), trials.tolist()):
            counts[participant_ids[s], own_acts[s, t], partner_acts[s, t], targets[s, t]] += 1

        self.log_probs = torch.log(counts / counts.sum(dim=-1, keepdim=True))
        return self

    def forward(self, xs: torch.Tensor, prev_state=None):
        participant_ids, own_acts, partner_acts, valid = self._unpack(xs)
        n_trials = xs.shape[1]
        logits = self.log_probs[
            participant_ids.unsqueeze(1).expand(-1, n_trials),
            own_acts,
            partner_acts,
        ]  # (B, T, N_ITEMS)
        logits[~valid] = float('nan')
        return logits.unsqueeze(2), None

    def count_parameters(self) -> int:
        # exactly one ape acts per event (bar 5 of 5905), so only the 2 * 3 reachable
        # (own, partner) cells carry data, each with N_ITEMS - 1 free probabilities
        return 2 * (N_ITEMS - 1) * (N_ITEMS - 1)
