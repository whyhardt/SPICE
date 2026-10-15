import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from spice import SpiceDataset, csv_to_dataset, split_data_along_blockdim

from weinhardt2026.utils.benchmarking_gru import GRUModel
from weinhardt2026.studies.kolff2025.spice_kolff2025 import CONFIG, N_ACTIONS, COL_TARGET_MASK, prepare_dataset, target_mask


# written by preprocessing_kolff2025.py (behavioural categories, see data/PREPROCESSING.md)
CATEGORIES_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025_categories.csv'
# modelling dataset, written by build_dataframe (both perspectives of every interaction)
DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025.csv'

CATEGORIES = ('Grooming_Solicitation', 'Self_Reposition', 'Directed_Scratch', 'Reposition_Body', 'Grooming_Process', 'Groom')
PARTNER_COLUMNS = tuple(f'partner_{category.lower()}' for category in CATEGORIES)
SELF_COLUMNS = tuple(f'self_{category.lower()}' for category in CATEGORIES)

DATA_KWARGS = dict(
    df_participant_id='participant',
    df_block='interaction_id',
    df_choice='choice',
    df_feedback=None,
    additional_inputs=CONFIG.additional_inputs,
)


# -------------------------------------------------------------------------------------------
# DATAFRAME: BOTH PERSPECTIVES OF EVERY INTERACTION
# -------------------------------------------------------------------------------------------

def build_dataframe(path_data: str = CATEGORIES_DATA_PATH) -> pd.DataFrame:
    """Every interaction twice: from the approached ape's (ID2's) and the approacher's (ID1's) point of view.

    ID1 and ID2 are taken from the interaction's first row; in two interactions some rows list the
    apes the other way round, so acts are assigned by ape name. Columns are self_<category> (the
    participant's own act) and partner_<category> (the other ape's act), one indicator per category.
    The target at event t is whether the participant grooms at event t+1 (`choice`, shifted by
    csv_to_dataset), defined only if it acts at t+1 (`self_acts_next`). Both perspectives of an
    interaction share its interaction_id, so a block split keeps them on the same side.

    Context columns (not model inputs, kept for analyses): `is_approacher` (1 = the participant is
    ID1), ranks normalized within community (hierarchies of depth 27 and 15), their difference
    centred within the participant and scaled to [-1, 1], and the own rank scaled to [-1, 1].
    """
    df = pd.read_csv(path_data)

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
    out['rank_own_scaled'] = 2 * out['rank_own'] - 1
    # readout: the participant grooms (1) or not (0); csv_to_dataset shifts it by one event
    out['choice'] = out['self_groom']
    out['self_acts_next'] = out.groupby(['participant', 'interaction_id'])['self_acts'].shift(-1).fillna(0.)
    return out


# -------------------------------------------------------------------------------------------
# HELD-OUT DYADS
# -------------------------------------------------------------------------------------------

def dyad_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> list[list[int]]:
    """Split the dyads into `n_folds` folds of whole dyads, balanced by number of targets.

    Dyads are visited in random order and each goes to the fold with the fewest targets so far.
    Returns the interaction ids per fold; testing on a fold asks whether the mechanism carries
    over to pairings the model was never fit on. Deterministic given `seed`.
    """
    bouts = df.groupby('interaction_id').agg(a=('participant', 'first'), b=('partner', 'first'), n_targets=('self_acts_next', 'sum'))
    bouts['dyad'] = [tuple(sorted(pair)) for pair in zip(bouts['a'], bouts['b'])]
    targets_per_dyad = bouts.groupby('dyad')['n_targets'].sum()

    rng = np.random.default_rng(seed)
    fold_of_dyad, load = {}, np.zeros(n_folds)
    for dyad in targets_per_dyad.index[rng.permutation(len(targets_per_dyad))]:
        fold = int(np.argmin(load))
        fold_of_dyad[dyad] = fold
        load[fold] += targets_per_dyad[dyad]

    return [sorted(int(i) for i, dyad in bouts['dyad'].items() if fold_of_dyad[dyad] == fold) for fold in range(n_folds)]


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

def get_dataset(
    path_data: str = DATA_PATH,
    test_fold: int = 0,
    n_folds: int = 5,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[SpiceDataset, SpiceDataset, dict]:
    """Load the modelling dataset and split off one fold of held-out dyads.

    The csv is built from the category data (build_dataframe) if it does not exist yet.
    ys = [not groom, groom, target mask] (see spice_kolff2025.prepare_dataset).
    """
    if not os.path.exists(path_data):
        build_dataframe().to_csv(path_data, index=False)
    df = pd.read_csv(path_data)

    test_blocks = dyad_folds(df, n_folds=n_folds, seed=seed)[test_fold]
    dataset = prepare_dataset(csv_to_dataset(file=df, **DATA_KWARGS))
    dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)

    info = {
        'n_participants': dataset.n_participants,
        'n_sessions_train': dataset_train.xs.shape[0],
        'n_sessions_test': dataset_test.xs.shape[0],
        'n_targets_train': int(target_mask(dataset_train.ys).sum()),
        'n_targets_test': int(target_mask(dataset_test.ys).sum()),
        'test_blocks': test_blocks,
    }
    if verbose:
        print({k: v for k, v in info.items() if k != 'test_blocks'})
    return dataset_train, dataset_test, info


def evaluation_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """ys reduced to [not groom, groom] and NaN where the participant does not act next.

    analysis_model_evaluation then scores only real targets when called with trial_filter=has_target.
    """
    ys = dataset.ys[..., :N_ACTIONS].clone()
    ys[~target_mask(dataset.ys)] = float('nan')
    return SpiceDataset(dataset.xs, ys, n_reward_features=0)


def has_target(ys: torch.Tensor) -> torch.Tensor:
    """trial_filter for analysis_model_evaluation on an evaluation_dataset."""
    return ~torch.isnan(ys[:, :, 0, 0])


# -------------------------------------------------------------------------------------------
# BENCHMARK: LAG-1 LOGISTIC REGRESSION (GROUP MODEL)
# -------------------------------------------------------------------------------------------

class LogisticRegressionModel(torch.nn.Module):
    """Baseline: logit P(groom next) = w . acts(t) + b, one model for all apes.

    The inputs are the 12 act indicators of the current event (the partner's and the participant's
    category), the same acts the SPICE model sees but without memory beyond the current event.
    Fitted on the scored targets only (near-unpenalized: C=1e4).
    """

    def __init__(self, C: float = 1e4):
        super().__init__()
        self.C = C
        self.columns = [N_ACTIONS + CONFIG.additional_inputs.index(c) for c in PARTNER_COLUMNS + SELF_COLUMNS]
        self.register_buffer('weight', torch.zeros(len(self.columns)))
        self.register_buffer('bias', torch.zeros(()))

    def _features(self, xs: torch.Tensor) -> torch.Tensor:
        return xs[:, :, 0, self.columns].nan_to_num(0.)

    def fit(self, dataset: SpiceDataset) -> 'LogisticRegressionModel':
        mask = target_mask(dataset.ys.cpu())[:, :, 0] & ~torch.isnan(dataset.xs.cpu()[:, :, 0, 0])
        X = self._features(dataset.xs.cpu())[mask].numpy()
        y = dataset.ys.cpu()[:, :, 0, 1][mask].numpy()
        clf = LogisticRegression(C=self.C, max_iter=10000).fit(X, y)
        self.weight = torch.tensor(clf.coef_[0], dtype=torch.float32)
        self.bias = torch.tensor(clf.intercept_[0], dtype=torch.float32)
        return self

    def forward(self, xs: torch.Tensor, prev_state=None):
        logit_groom = self._features(xs) @ self.weight + self.bias                        # (B, T)
        logits = torch.stack((torch.zeros_like(logit_groom), logit_groom), dim=-1)         # (B, T, 2)
        logits[torch.isnan(xs[:, :, 0, 0])] = float('nan')
        return logits.unsqueeze(2), None

    def count_parameters(self) -> int:
        return len(self.columns) + 1


# -------------------------------------------------------------------------------------------
# GRU WITH THE SPICE MODEL'S INPUTS
# -------------------------------------------------------------------------------------------

class ModelInputGRU(GRUModel):
    """GRU (no participant embedding) that sees what the SPICE model sees.

    The input columns are CONFIG.additional_inputs without the last one (the target mask, which
    is future information); the choice input (a duplicate of the participant's own grooming) is zeroed.
    """

    def __init__(self, hidden_size: int = 8, **kwargs):
        super().__init__(n_actions=N_ACTIONS, additional_inputs=len(CONFIG.additional_inputs) - 1,
                         n_reward_features=0, hidden_size=hidden_size, **kwargs)

    def forward(self, inputs, state=None):
        inputs = inputs.clone()
        inputs[..., :self.n_actions] = 0.
        return super().forward(inputs, state)


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING: CLOSED LOOP ON THE PARTICIPANT'S OWN GROOMING
# -------------------------------------------------------------------------------------------

COLUMNS_SELF = [N_ACTIONS + CONFIG.additional_inputs.index(c) for c in SELF_COLUMNS]
INDEX_SELF_GROOM = SELF_COLUMNS.index('self_groom')


@torch.no_grad()
def generate_behavior(model, dataset: SpiceDataset) -> SpiceDataset:
    """Simulate the participant's grooming in replayed interactions.

    The models predict only whether the participant's next act is grooming, so only that decision is
    closed-loop: the partner's acts and the timing of the participant's acts (whether it acts at an
    event, `self_acts_next`) are replayed from the real interaction. Wherever the participant acts at
    event t+1, its act is drawn from the model's P(groom) after event t: grooming, or else its real
    act if that was not grooming, or a non-grooming category drawn from the participants' observed
    non-grooming acts if it was. The participant's act at the first event is replayed. Sampled acts
    feed back into the model's inputs (self_*) and the choice columns.

    Args:
        model: Fitted model (SpiceEstimator or torch.nn.Module) returning grooming logits.
        dataset: Original dataset (prepare_dataset layout).

    Returns:
        SpiceDataset with the simulated self_* acts in xs and the simulated targets in ys.
    """
    tensors = [] if hasattr(model, 'device') else list(model.parameters()) + list(model.buffers())
    device = model.device if hasattr(model, 'device') else (tensors[0].device if tensors else torch.device('cpu'))
    n_sessions, n_trials = dataset.xs.shape[:2]
    valid = ~torch.isnan(dataset.xs[:, :, 0, 0])
    acts_real = dataset.xs[:, :, 0, COLUMNS_SELF].nan_to_num(0.)          # (S, T, 6) one-hot, 0 = no act
    acts_next = dataset.xs[:, :, 0, COL_TARGET_MASK].nan_to_num(0.) == 1    # participant acts at t+1

    # distribution of the observed non-grooming acts
    counts = acts_real[valid].sum(dim=0)
    counts[INDEX_SELF_GROOM] = 0.
    p_other = counts / counts.sum()

    xs_sim = dataset.xs.clone()
    ys_sim = dataset.ys.clone()
    acts = acts_real[:, 0].clone()
    state = None
    for t in range(n_trials):
        xs_t = xs_sim[:, t].clone()
        xs_t[:, 0, COLUMNS_SELF] = acts
        groom = acts[:, INDEX_SELF_GROOM]
        xs_t[:, 0, :N_ACTIONS] = torch.stack((1 - groom, groom), dim=-1)
        xs_t[~valid[:, t]] = 0.
        xs_sim[:, t] = xs_t

        logits, state = model(xs_t[:, None].to(device), state)
        if logits.dim() == 5:                                    # BaseModel: (E, B, T, W, A)
            logits = logits.mean(dim=0)
        p_groom = torch.softmax(logits[:, -1, 0, :N_ACTIONS].float().cpu(), dim=-1)[:, 1].nan_to_num(0.5)
        groom_next = torch.bernoulli(p_groom)

        if t + 1 < n_trials:
            real_next = acts_real[:, t + 1]
            other = torch.nn.functional.one_hot(torch.multinomial(p_other, n_sessions, replacement=True), len(SELF_COLUMNS)).float()
            not_groom = torch.where(real_next[:, INDEX_SELF_GROOM:INDEX_SELF_GROOM + 1] == 1, other, real_next)
            acts = torch.where(groom_next[:, None] == 1, torch.nn.functional.one_hot(torch.tensor(INDEX_SELF_GROOM), len(SELF_COLUMNS)).float(), not_groom)
            acts = acts * acts_next[:, t, None]
            ys_sim[:, t, 0, :N_ACTIONS] = torch.stack((1 - acts[:, INDEX_SELF_GROOM], acts[:, INDEX_SELF_GROOM]), dim=-1)

    xs_sim[~valid] = float('nan')
    ys_sim[~valid] = float('nan')
    return SpiceDataset(xs_sim, ys_sim, n_reward_features=0)
