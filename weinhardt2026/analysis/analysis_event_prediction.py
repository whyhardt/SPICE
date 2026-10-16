"""
Event prediction: how well each model predicts the rare decisions of a task (leave, switch, groom)
that the overall one-step likelihood under-weights, since it is dominated by the frequent default
behavior (harvest, stay, repeat).

Teacher-forced: every model sees the real history and predicts P(event) on every trial with a next
action. Per model:
    auc                 ROC AUC of P(event) for the events (headline; rank-based, cannot be gamed by
                        inflating P(event))
    average_precision   area under the precision-recall curve
    event_rate          observed rate of events; mean_p_event is the mean predicted P(event)
                        (calibration in the large: they should agree)
    nll_event           mean -log P(event) on event trials
    nll_no_event        mean -log P(no event) on the other trials
    auc_participant     mean of the per-participant AUCs
Every model: one model per final seed (SPICE: the ensemble member with the highest training likelihood),
as in the generative comparison; scores are averaged over seeds (`*_sd` = SD over seeds). Per participant,
SPICE-EQ's AUC (mean over seeds) is compared with the benchmark's by a two-sided sign test.

Events per study are registered in EVENTS. Data and models come from the STUDIES registry of
weinhardt2026/figures/figure_model_evaluation.py, on the train/test split the models were fitted on.

Outputs in weinhardt2026/studies/<study>/results/:
    event_prediction.csv                 per split and model
    event_prediction_participants.csv    per split, model and participant: AUC and n events
    event_prediction.png/.pdf            ROC curves per split; per-participant AUC SPICE-EQ vs benchmark

Usage:
    python weinhardt2026/analysis/analysis_event_prediction.py bustamante2023
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules import 'studies.<study>...' (as in run.py)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from spice import SpiceDataset
from weinhardt2026.analysis.analysis_generative_comparison import (
    COLORS, EnsembleMember, load_registered_models, study_dir,
)


EPS = 1e-9
MODELS = ['GRU', 'Benchmark', 'SPICE-EQ', 'SPICE-RNN']

# Per study: the event's name and the action index whose choice at t+1 is the event.
EVENTS = {
    'bustamante2023': dict(name='Leave', action=1),   # 0 = harvest, 1 = exit the patch
}


# -------------------------------------------------------------------------------------------
# PREDICTIONS
# -------------------------------------------------------------------------------------------

@torch.no_grad()
def predict_probabilities(model, dataset: SpiceDataset, use_sindy: bool = None) -> np.ndarray:
    """Teacher-forced action probabilities (B, T, A); `use_sindy` only for SPICE (EnsembleMember)."""
    if isinstance(model, EnsembleMember):
        model.eval(use_sindy=use_sindy)
        model.estimator.model.init_state(batch_size=dataset.xs.shape[0])
        outputs, _ = model(dataset.xs.to(model.device))
        outputs = outputs[0]
    else:
        model.eval()
        outputs, _ = model.to('cpu')(dataset.xs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    logits = outputs.detach().float().cpu()[:, :, 0, :dataset.n_actions]
    return torch.softmax(logits, dim=-1).numpy()


def event_targets(dataset: SpiceDataset, action: int) -> tuple[np.ndarray, np.ndarray]:
    """events (B, T): the next action is `action`; valid (B, T): trials with an input and a next action."""
    xs, ys = dataset.xs.numpy(), dataset.ys.numpy()
    valid = ~np.isnan(xs[:, :, 0, 0]) & ~np.isnan(ys[:, :, 0, 0])
    events = np.nan_to_num(ys[:, :, 0, action]) == 1
    return events, valid


# -------------------------------------------------------------------------------------------
# SCORES
# -------------------------------------------------------------------------------------------

def _auc(events: np.ndarray, p_event: np.ndarray) -> float:
    return roc_auc_score(events, p_event) if 0 < events.sum() < len(events) else np.nan


def score_events(p_event: np.ndarray, events: np.ndarray, valid: np.ndarray, participant_ids: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Pooled event scores and per-participant AUCs for one model."""
    p, e = p_event[valid], events[valid]
    scores = dict(
        auc=_auc(e, p),
        average_precision=average_precision_score(e, p),
        event_rate=e.mean(),
        mean_p_event=p.mean(),
        nll_event=-np.log(np.clip(p[e], EPS, 1)).mean(),
        nll_no_event=-np.log(np.clip(1 - p[~e], EPS, 1)).mean(),
    )
    sessions_participant = np.broadcast_to(participant_ids[:, None], valid.shape)[valid]
    rows = [
        dict(participant_id=int(pid), auc=_auc(e[mask], p[mask]), n_events=int(e[mask].sum()))
        for pid in np.unique(sessions_participant) for mask in [sessions_participant == pid]
    ]
    participants = pd.DataFrame(rows)
    scores['auc_participant'] = participants['auc'].mean()
    return scores, participants


def compare_participants(participants: pd.DataFrame, model: str, reference: str) -> dict:
    """Sign test across participants: how often `model`'s AUC exceeds `reference`'s (ties dropped)."""
    wide = participants.pivot(index='participant_id', columns='model', values='auc')[[model, reference]].dropna()
    difference = wide[model] - wide[reference]
    n_better, n_worse = int((difference > 0).sum()), int((difference < 0).sum())
    p_value = binomtest(n_better, n_better + n_worse).pvalue if n_better + n_worse > 0 else np.nan
    return dict(model=model, reference=reference, n_better=n_better, n_worse=n_worse, p_sign_test=p_value,
                median_difference=difference.median())


# -------------------------------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------------------------------

def run_event_prediction(study: str) -> pd.DataFrame:
    from weinhardt2026.figures.figure_model_evaluation import load_study
    event = EVENTS[study]
    _, dataset_train, dataset_test = load_study(study)
    models = load_registered_models(study, dataset_train)

    score_rows, participant_rows, roc = [], [], {}
    for split, dataset in (('test', dataset_test), ('train', dataset_train)):
        events, valid = event_targets(dataset, event['action'])
        participant_ids = dataset.xs[:, 0, 0, -1].long().numpy()
        for name in MODELS:
            instances = models['SPICE' if name.startswith('SPICE') else name]
            instance_scores, instance_participants, instance_p = [], [], []
            for model in instances:
                use_sindy = (name == 'SPICE-EQ') if name.startswith('SPICE') else None
                p_event = predict_probabilities(model, dataset, use_sindy)[..., event['action']]
                scores, participants = score_events(p_event, events, valid, participant_ids)
                instance_scores.append(scores)
                instance_participants.append(participants.set_index('participant_id'))
                instance_p.append(p_event[valid])
            table = pd.DataFrame(instance_scores)
            row = dict(split=split, model=name, n_models=len(instances), n_trials=int(valid.sum()), n_events=int(events[valid].sum()))
            for column in table:
                row[column] = table[column].mean()
                if len(instances) > 1:
                    row[f'{column}_sd'] = table[column].std()
            score_rows.append(row)
            participants = pd.concat(instance_participants).groupby(level=0).agg(auc=('auc', 'mean'), n_events=('n_events', 'first'))
            participant_rows.append(participants.reset_index().assign(split=split, model=name))
            roc[(split, name)] = roc_curve(events[valid], np.mean(instance_p, axis=0))[:2]

    scores = pd.DataFrame(score_rows)
    participants = pd.concat(participant_rows, ignore_index=True)
    tests = pd.DataFrame([
        dict(split=split, **compare_participants(participants[participants['split'] == split], 'SPICE-EQ', 'Benchmark'))
        for split in ('test', 'train')
    ])

    output_dir = study_dir(study, 'results')
    os.makedirs(output_dir, exist_ok=True)
    scores.to_csv(os.path.join(output_dir, 'event_prediction.csv'), index=False)
    participants[['split', 'model', 'participant_id', 'auc', 'n_events']].to_csv(
        os.path.join(output_dir, 'event_prediction_participants.csv'), index=False)
    tests.to_csv(os.path.join(output_dir, 'event_prediction_sign_test.csv'), index=False)
    plot_event_prediction(roc, scores, participants, tests, event['name'], study, os.path.join(output_dir, 'event_prediction'))

    columns = ['split', 'model', 'auc', 'average_precision', 'event_rate', 'mean_p_event', 'nll_event', 'nll_no_event', 'auc_participant']
    print(f"\n{study}: {event['name']} prediction (teacher-forced)")
    print(scores[columns].round(3).to_string(index=False))
    print('\nPer-participant AUC, SPICE-EQ vs Benchmark (sign test):')
    print(tests.round(4).to_string(index=False))
    print(f'Saved to {output_dir}')
    return scores


# -------------------------------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------------------------------

def plot_event_prediction(roc: dict, scores: pd.DataFrame, participants: pd.DataFrame, tests: pd.DataFrame,
                          event_name: str, title: str, path: str):
    """ROC curves for the held-out and training data; per-participant held-out AUC, SPICE-EQ vs benchmark."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    score = scores.set_index(['split', 'model'])
    for ax, split in zip(axes[:2], ('test', 'train')):
        for name in MODELS:
            fpr, tpr = roc[(split, name)]
            ax.plot(fpr, tpr, color=COLORS[name], linewidth=2, linestyle='--' if name == 'SPICE-RNN' else '-',
                    label=f"{name}  AUC {score.loc[(split, name), 'auc']:.3f}")
        ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=1)
        ax.set_title(f"{event_name} prediction ({'held-out' if split == 'test' else 'training'} blocks)", fontsize=10)
        ax.set_xlabel('False positive rate', fontsize=9)
        ax.set_ylabel('True positive rate', fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc='lower right')

    ax = axes[2]
    test = participants[participants['split'] == 'test'].pivot(index='participant_id', columns='model', values='auc')
    paired = test[['Benchmark', 'SPICE-EQ']].dropna()
    ax.scatter(paired['Benchmark'], paired['SPICE-EQ'], s=12, color=COLORS['SPICE-EQ'], alpha=0.6, linewidths=0)
    low = np.nanmin(paired.to_numpy()) - 0.02
    ax.plot([low, 1], [low, 1], color='#bbbbbb', linewidth=1)
    sign = tests.set_index('split').loc['test']
    ax.set_title(f"Per participant (held-out): SPICE-EQ better in {sign['n_better']} / {sign['n_better'] + sign['n_worse']}\n"
                 f"sign test p = {sign['p_sign_test']:.2g}", fontsize=10)
    ax.set_xlabel('Benchmark AUC', fontsize=9)
    ax.set_ylabel('SPICE-EQ AUC', fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(f'{path}.{extension}', bbox_inches='tight', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teacher-forced event prediction (AUC) per model.')
    parser.add_argument('studies', nargs='+', help=f"study names ({', '.join(EVENTS)})")
    args = parser.parse_args()
    for study in args.studies:
        run_event_prediction(study)
