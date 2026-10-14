import numpy as np
import pandas as pd
import torch

from spice import SpiceDataset, csv_to_dataset, split_data_along_blockdim

from weinhardt2026.studies.kolff2025.spice_kolff2025_groom import CONFIG


DEFAULT_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_original.csv'
PERSPECTIVE_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025.csv'

# signalling-act alphabet of the raw data (the *control signal* alphabet)
N_SIGNAL_ACTS = 5
ACT_ACTION, ACT_GROOMING, ACT_GESTURE, ACT_SCRATCHING, ACT_WAITING = range(N_SIGNAL_ACTS)

# outcome alphabet predicted by the model (the *readout* alphabet)
N_OUTCOMES = 3
OUTCOME_NEGOTIATE, OUTCOME_OWNGROOM, OUTCOME_PARTNERGROOM = range(N_OUTCOMES)
OUTCOME_NAMES = {
    OUTCOME_NEGOTIATE: 'negotiate',
    OUTCOME_OWNGROOM: 'owngroom',
    OUTCOME_PARTNERGROOM: 'partnergroom',
}


# -------------------------------------------------------------------------------------------
# PERSPECTIVE RESHAPING
# -------------------------------------------------------------------------------------------

def build_perspective_dataframe(path_data: str = None) -> pd.DataFrame:
    """Re-encode the dyadic event stream into one sequence per (focal ape, interaction).

    The raw CSV stores every event of a dyadic bout once, with the roles ID1/ID2 assigned
    inconsistently across rows (the actor is sometimes ID1, sometimes ID2). Keeping only
    the rows where a given ape happens to be ID1 therefore drops roughly half of what that
    ape observed, and biases what it appears to have done relative to what it received.
    Here every interaction is written out twice — once from each ape's point of view — so
    that a focal ape sees the complete time-ordered stream of its own acts and the acts it
    received.

    Dominance ranks are normalized *within community*: the two communities have hierarchies
    of different depth (0-26 and 0-14), so a one-step rank gap does not mean the same thing
    in both. Normalizing globally would shrink the small community's rank distances by half.

    Returns one row per (focal ape, event), with columns:
        focal, partner, interaction_id, start, own_action, partner_action, outcome,
        rank_own, rank_partner, community_id
    """

    df = pd.read_csv(path_data if path_data is not None else DEFAULT_DATA_PATH)

    # per-community rank normalization -> [0, 1] within each hierarchy
    ranks_long = pd.concat([
        df[['community_id', 'Dominance_rank_ID1']].rename(columns={'Dominance_rank_ID1': 'rank'}),
        df[['community_id', 'Dominance_rank_ID2']].rename(columns={'Dominance_rank_ID2': 'rank'}),
    ])
    max_rank = ranks_long.groupby('community_id')['rank'].max().to_dict()

    records = []
    for interaction_id, bout in df.groupby('interaction_id'):
        bout = bout.sort_values('start')
        apes = sorted(set(bout['ID1']) | set(bout['ID2']))
        if len(apes) != 2:
            continue  # skip anything that is not strictly dyadic
        for focal in apes:
            partner = apes[1] if apes[0] == focal else apes[0]
            for _, event in bout.iterrows():
                if event['ID1'] == focal:
                    own_act, partner_act = event['SigAct_ID1'], event['SigAct_ID2']
                    rank_own, rank_partner = event['Dominance_rank_ID1'], event['Dominance_rank_ID2']
                elif event['ID2'] == focal:
                    own_act, partner_act = event['SigAct_ID2'], event['SigAct_ID1']
                    rank_own, rank_partner = event['Dominance_rank_ID2'], event['Dominance_rank_ID1']
                else:
                    continue
                scale = max_rank[event['community_id']]
                records.append({
                    'focal': focal,
                    'partner': partner,
                    'interaction_id': interaction_id,
                    'start': event['start'],
                    'own_action': int(own_act),
                    'partner_action': int(partner_act),
                    'outcome': _outcome(own_act, partner_act),
                    'rank_own': rank_own / scale,
                    'rank_partner': rank_partner / scale,
                    'community_id': event['community_id'],
                })

    df_perspective = pd.DataFrame.from_records(records)
    df_perspective['rank_diff'] = df_perspective['rank_own'] - df_perspective['rank_partner']
    return df_perspective.sort_values(['focal', 'interaction_id', 'start']).reset_index(drop=True)


def _outcome(own_act: int, partner_act: int) -> int:
    """Collapse a pair of concurrent signalling acts into the 3-way readout class.

    Simultaneous grooming in both directions occurs in 0.03% of events and is resolved
    focal-centrically to `owngroom`.
    """
    if own_act == ACT_GROOMING:
        return OUTCOME_OWNGROOM
    if partner_act == ACT_GROOMING:
        return OUTCOME_PARTNERGROOM
    return OUTCOME_NEGOTIATE


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

def get_dataset(
    path_data: str = None,
    test_fraction: float = 0.2,
    min_length: int = 5,
    seed: int = 42,
    save_csv: str = PERSPECTIVE_DATA_PATH,
) -> tuple[SpiceDataset, SpiceDataset, dict]:
    """Build the grooming-expectation dataset.

    One session = one (focal ape, interaction) perspective. The target at trial t is the
    one-hot outcome at t+1 over [negotiate, owngroom, partnergroom]. Held-out data is a
    random subset of *interactions*, so both perspectives of a bout stay on the same side
    of the split and the test set contains bouts the model has never seen.
    """

    df = build_perspective_dataframe(path_data)

    # drop bouts too short to carry dynamics (a single event yields zero targets)
    lengths = df.groupby(['focal', 'interaction_id'])['start'].transform('size')
    df = df[lengths >= min_length].copy()

    # Within-ape centring of the dominance distance.
    #
    # Raw rank_diff is nearly a restatement of the focal ape's own rank: averaged over an
    # ape's partners it collapses to rank_own minus a near-constant, and empirically the
    # two correlate at rho = +0.92. Regressing a coefficient-on-rank_diff against rank_own
    # would then be close to circular. Subtracting each ape's own mean cancels the
    # f(rank_own) term exactly, leaving only "is this partner more or less dominant than my
    # typical partner" -- which is orthogonal to rank_own, and to every other per-ape
    # variable, by construction (the centred values sum to zero within each ape).
    #
    # Centred here rather than in build_perspective_dataframe so the mean is taken over the
    # events actually fitted, making that orthogonality exact rather than approximate.
    # Apes with a single partner get rank_diff_centered == 0 throughout and therefore carry
    # no information about this term; exclude them when interpreting its coefficient.
    df['rank_diff_centered'] = (
        df['rank_diff'] - df.groupby('focal')['rank_diff'].transform('mean')
    )

    if save_csv:
        df.to_csv(save_csv, index=False)

    interactions = np.sort(df['interaction_id'].unique())
    rng = np.random.default_rng(seed)
    n_test = int(round(test_fraction * len(interactions)))
    test_blocks = [int(i) for i in rng.choice(interactions, size=n_test, replace=False)]

    dataset = csv_to_dataset(
        file=df,
        df_participant_id='focal',
        df_block='interaction_id',
        df_choice='outcome',
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
        'outcome_frequencies': {
            OUTCOME_NAMES[k]: round(float(v), 4)
            for k, v in df['outcome'].value_counts(normalize=True).sort_index().items()
        },
        'n_test_interactions': len(test_blocks),
    }

    return dataset_train, dataset_test, info_dataset


# -------------------------------------------------------------------------------------------
# BENCHMARK: CONDITIONAL FREQUENCY MODEL
# -------------------------------------------------------------------------------------------

class ConditionalGroomingModel(torch.nn.Module):
    """Baseline: P(outcome at t+1 | own act at t, received act at t), per focal ape.

    A memoryless lookup table — no latent expectation, no carry-over across events.
    Whatever SPICE gains over this is attributable to the latent dynamics.
    """

    def __init__(self, n_participants: int, smoothing: float = 1.0):
        super().__init__()
        self.n_participants = n_participants
        self.smoothing = smoothing
        self.register_buffer(
            'log_probs',
            torch.zeros(n_participants, N_SIGNAL_ACTS, N_SIGNAL_ACTS, N_OUTCOMES),
        )

    @staticmethod
    def _unpack(xs: torch.Tensor):
        # feature layout: [outcome one-hot (3), own_action, partner_action,
        #                  rank_own, rank_partner, 5 metadata cols]
        participant_ids = xs[:, 0, 0, -1].nan_to_num(0).long()
        own_actions = xs[:, :, 0, N_OUTCOMES].nan_to_num(ACT_WAITING).long().clamp(0, N_SIGNAL_ACTS - 1)
        partner_actions = xs[:, :, 0, N_OUTCOMES + 1].nan_to_num(ACT_WAITING).long().clamp(0, N_SIGNAL_ACTS - 1)
        valid = ~torch.isnan(xs[:, :, 0, 0])
        return participant_ids, own_actions, partner_actions, valid

    def fit(self, dataset: SpiceDataset) -> 'ConditionalGroomingModel':
        xs, ys = dataset.xs.cpu(), dataset.ys.cpu()
        participant_ids, own_actions, partner_actions, valid = self._unpack(xs)
        valid = valid & ~torch.isnan(ys[:, :, 0, 0])
        targets = ys[:, :, 0, :].nan_to_num(0).argmax(dim=-1)

        counts = torch.full(
            (self.n_participants, N_SIGNAL_ACTS, N_SIGNAL_ACTS, N_OUTCOMES), self.smoothing,
        )
        sessions, trials = torch.nonzero(valid, as_tuple=True)
        for s, t in zip(sessions.tolist(), trials.tolist()):
            counts[participant_ids[s], own_actions[s, t], partner_actions[s, t], targets[s, t]] += 1

        self.log_probs = torch.log(counts / counts.sum(dim=-1, keepdim=True))
        return self

    def forward(self, xs: torch.Tensor, prev_state=None):
        participant_ids, own_actions, partner_actions, valid = self._unpack(xs)
        n_trials = xs.shape[1]
        logits = self.log_probs[
            participant_ids.unsqueeze(1).expand(-1, n_trials),
            own_actions,
            partner_actions,
        ]  # (B, T, N_OUTCOMES)
        logits[~valid] = float('nan')
        return logits.unsqueeze(2), None

    def count_parameters(self) -> int:
        # own and partner act never co-occur, so only the 2 * 4 + 1 reachable
        # (own, partner) cells carry data, each with N_OUTCOMES - 1 free probabilities
        return (2 * (N_SIGNAL_ACTS - 1) + 1) * (N_OUTCOMES - 1)
