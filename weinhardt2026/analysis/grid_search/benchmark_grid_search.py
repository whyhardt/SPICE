"""Grid search over the training hyperparameters of the benchmark models and the GRU baselines.

One call fits one grid point of one study for all requested seeds, either
- of the GRU baseline (--kind gru: lr x scheduler x hidden_size x dropout), or
- of the study's benchmark model (--kind benchmark: lr x scheduler).

Each study is set up like its run script (weinhardt2026/studies/<study>/<study>.py): same data loading,
test split, loss, benchmark constructor and GRU inputs (no embeddings); everything trains on the GPU
with the test set passed to training() (the lr scheduler steps on the test loss). Model selection
happens on the test loss (see run_benchmark_grid_search.py).

The benchmark models start from fixed parameter values and train deterministically, so each seed
draws its own random start instead: every parameter (all are unconstrained raw/log values) is
jittered around its default by N(0, --init_std).

Every fitted model is saved to <out_dir>/<study>/<name>.pkl next to <name>.json with its config and
train/test loss. Seeds with an existing json are skipped, so a timed-out array task can simply be resubmitted.

Usage (from the repository root):
    python weinhardt2026/analysis/grid_search/benchmark_grid_search.py --study dezfouli2019 --kind gru \
        --lr 0.01 --scheduler 1 --hidden_size 8 --dropout 0.1 --seeds 0 1 2
"""

import sys
import argparse
import json
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from spice import SpiceDataset, cross_entropy_loss, split_data_along_blockdim
from weinhardt2026.utils.benchmarking_gru import GRUModel, training


DEFAULT_OUT_DIR = 'weinhardt2026/analysis/grid_search/runs'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class StudySetup:
    """Everything needed to fit the benchmark and the GRU of one study.

    make_benchmark: () -> model, or None if the study's benchmark is not fitted by gradient descent.
    make_gru: (hidden_size, dropout) -> model.
    """
    dataset_train: SpiceDataset
    dataset_test: SpiceDataset
    make_gru: Callable[[int, float], torch.nn.Module]
    make_benchmark: Optional[Callable[[], torch.nn.Module]] = None
    loss_fn: Callable = cross_entropy_loss
    loss_kwargs: dict = field(default_factory=dict)


# -------------------------------------------------------------------------------------------
# PER-STUDY SETUPS (mirroring weinhardt2026/studies/<study>/<study>.py)
# -------------------------------------------------------------------------------------------

def setup_braun2018(path_data: str = 'weinhardt2026/studies/braun2018/data/braun2018.csv') -> StudySetup:
    from weinhardt2026.studies.braun2018.benchmarking_braun2018 import ExpectedValueControl, get_dataset

    dataset_train, dataset_test, _ = get_dataset(path_data=path_data, test_blocks=(3, 6, 9))

    # Truncate outlier-length blocks (mean + 2*std)
    n_trials_per_block = (~dataset_train.xs[:, :, 0, 0].isnan()).sum(dim=1).float()
    cutoff = int(n_trials_per_block.mean() + 2 * n_trials_per_block.std(unbiased=False))
    for dataset in (dataset_train, dataset_test):
        dataset.xs = dataset.xs[:, :cutoff]
        dataset.ys = dataset.ys[:, :cutoff]

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: ExpectedValueControl(n_participants=dataset_train.n_participants),
        make_gru=lambda hidden_size, dropout: GRUModel(dataset_train.n_actions, additional_inputs=4, n_reward_features=0, hidden_size=hidden_size, dropout=dropout),
    )


def setup_bruckner2025(path_data: str = 'weinhardt2026/studies/bruckner2025/data/bruckner2025.csv') -> StudySetup:
    from weinhardt2026.studies.bruckner2025.spice_bruckner2025 import mse_loss
    from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import RationalResourceModel, get_dataset

    dataset, _ = get_dataset(path_data=path_data)
    dataset_train, dataset_test = split_data_along_blockdim(dataset, (2, 4))

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: RationalResourceModel(n_participants=dataset_train.n_participants),
        make_gru=lambda hidden_size, dropout: GRUModel(n_actions=1, additional_inputs=6, n_reward_features=dataset_train.n_reward_features, hidden_size=hidden_size, dropout=dropout),
        loss_fn=mse_loss,
    )


def setup_bustamante2023(path_data: str = 'weinhardt2026/studies/bustamante2023/data/bustamante2023.csv') -> StudySetup:
    from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import EffortCostMVTModel, get_dataset

    dataset_train, dataset_test, _ = get_dataset(path_data=path_data, test_blocks=(3, 6))

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: EffortCostMVTModel(n_participants=dataset_train.n_participants, path_data=path_data, batch_first=True),
        # harvest_duration, travel_duration (constant, scaled to [0, 1]), is_msit, is_high
        make_gru=lambda hidden_size, dropout: GRUModel(dataset_train.n_actions, additional_inputs=4, additional_inputs_scale=(10., 10., 1., 1.), hidden_size=hidden_size, dropout=dropout),
    )


def setup_dezfouli2019(path_data: str = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv') -> StudySetup:
    from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import GQLModel, get_dataset

    dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=(3, 6, 9))

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: GQLModel(n_participants=info_dataset['n_participants'], batch_first=True),
        make_gru=lambda hidden_size, dropout: GRUModel(n_actions=info_dataset['n_actions'], additional_inputs=2, hidden_size=hidden_size, dropout=dropout),
    )


def setup_eckstein2026(path_data: str = 'weinhardt2026/studies/eckstein2026/data/eckstein2026.csv') -> StudySetup:
    from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import BestRLModel, get_dataset

    dataset_train, dataset_test, _ = get_dataset(path_data=path_data, test_blocks=(2,))

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: BestRLModel(n_participants=dataset_train.n_participants, n_actions=dataset_train.n_actions, batch_first=True),
        make_gru=lambda hidden_size, dropout: GRUModel(n_actions=dataset_train.n_actions, additional_inputs=2, hidden_size=hidden_size, dropout=dropout),
        loss_kwargs={'label_smoothing': 0.0},
    )


def setup_ganesh2024a(path_data: str = 'weinhardt2026/studies/ganesh2024a/data/ganesh2024a.csv') -> StudySetup:
    from weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a import BayesianModel, get_dataset

    dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=(3, 6, 9))

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: BayesianModel(n_participants=info_dataset['n_participants'], batch_first=True),
        make_gru=lambda hidden_size, dropout: GRUModel(n_actions=info_dataset['n_actions'], additional_inputs=2, hidden_size=hidden_size, dropout=dropout),
    )


def setup_kolff2025() -> StudySetup:
    # the benchmark (LogisticRegressionModel) is fitted in closed form -> nothing to search
    from weinhardt2026.studies.kolff2025.spice_kolff2025 import LOSS_FN
    from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import ModelInputGRU, get_dataset

    dataset_train, dataset_test, _ = get_dataset(test_fold=0)

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_gru=lambda hidden_size, dropout: ModelInputGRU(hidden_size=hidden_size, dropout=dropout),
        loss_fn=LOSS_FN,
    )


def setup_weber2024(path_data: str = 'weinhardt2026/studies/weber2024/data/weber2024.csv') -> StudySetup:
    from weinhardt2026.studies.weber2024.spice_weber2024 import clamped_angular_mse
    from weinhardt2026.studies.weber2024.benchmarking_weber2024 import ChangePointModel, get_dataset

    dataset, _ = get_dataset(path_data=path_data)
    dataset_train, dataset_test = split_data_along_blockdim(dataset, (3, 5, 9, 12))

    return StudySetup(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        make_benchmark=lambda: ChangePointModel(n_participants=dataset_train.n_participants, h_step=2, variance_inference=True),
        make_gru=lambda hidden_size, dropout: GRUModel(
            n_actions=2,
            additional_inputs=4,
            additional_inputs_scale=(1., 1., 1., 90.),
            n_reward_features=dataset_train.n_reward_features,
            hidden_size=hidden_size,
            dropout=dropout,
        ),
        loss_fn=clamped_angular_mse,
    )


# setup(path_data) with the study's own CSV as default (model identification passes simulated ones)
STUDIES: dict[str, Callable[..., StudySetup]] = {
    'braun2018': setup_braun2018,
    'bruckner2025': setup_bruckner2025,
    'bustamante2023': setup_bustamante2023,
    'dezfouli2019': setup_dezfouli2019,
    'eckstein2026': setup_eckstein2026,
    'ganesh2024a': setup_ganesh2024a,
    'kolff2025': setup_kolff2025,
    'weber2024': setup_weber2024,
}


# -------------------------------------------------------------------------------------------
# FITTING
# -------------------------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def randomize_parameters(model: torch.nn.Module, std: float) -> torch.nn.Module:
    """Random start around the default init: adds N(0, std) to every (unconstrained) parameter."""
    for parameter in model.parameters():
        parameter.add_(torch.randn_like(parameter) * std)
    return model


@torch.no_grad()
def compute_loss(model: torch.nn.Module, dataset: SpiceDataset, loss_fn: Callable, loss_kwargs: dict) -> float:
    """Loss on a whole dataset, masked like the test loss in benchmarking_gru.training."""
    model.eval()
    xs, ys = dataset.xs.to(DEVICE), dataset.ys.to(DEVICE)
    logits, _ = model(xs)
    nan_mask = ~torch.isnan(xs[..., :model.n_actions].sum(dim=-1))
    return float(loss_fn(logits[nan_mask], ys[nan_mask], **loss_kwargs).item())


def run_name(kind: str, study: str, lr: float, scheduler: bool, seed: int, hidden_size: int = None, dropout: float = None) -> str:
    name = f"{kind}_{study}_lr{lr}_sc{int(scheduler)}"
    if kind == 'gru':
        name += f"_hs{hidden_size}_do{dropout}"
    return name + f"_rs{seed}"


def fit_one(setup: StudySetup, args: argparse.Namespace, seed: int) -> tuple[torch.nn.Module, dict]:
    set_seeds(seed)
    if args.kind == 'gru':
        model = setup.make_gru(args.hidden_size, args.dropout)
    else:
        model = randomize_parameters(setup.make_benchmark(), args.init_std)

    start = time.time()
    model = training(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        dataset_train=setup.dataset_train,
        dataset_test=setup.dataset_test,
        epochs=args.epochs,
        loss_fn=setup.loss_fn,
        loss_kwargs=setup.loss_kwargs,
        scheduler=bool(args.scheduler),
        device=DEVICE,
    )
    duration = time.time() - start

    metrics = {
        'train_loss': compute_loss(model, setup.dataset_train, setup.loss_fn, setup.loss_kwargs),
        'test_loss': compute_loss(model, setup.dataset_test, setup.loss_fn, setup.loss_kwargs),
        'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'duration_sec': duration,
    }
    return model, metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.study
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {seed: run_name(args.kind, args.study, args.lr, args.scheduler, seed, args.hidden_size, args.dropout) for seed in args.seeds}
    seeds = [seed for seed in args.seeds if not (out_dir / f"{names[seed]}.json").exists()]
    if not seeds:
        print(f"[{args.study}] all seeds exist, nothing to do")
        return

    setup = STUDIES[args.study]()
    if args.kind == 'benchmark' and setup.make_benchmark is None:
        print(f"[{args.study}] no trainable benchmark, nothing to do")
        return

    failed = []
    for seed in seeds:
        name = names[seed]
        print(f"[{args.study}] fitting {name}")
        try:
            model, metrics = fit_one(setup, args, seed)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue

        config = {
            'study': args.study,
            'kind': args.kind,
            'lr': args.lr,
            'epochs': args.epochs,
            'scheduler': bool(args.scheduler),
            'hidden_size': args.hidden_size,
            'dropout': args.dropout,
            'init_std': args.init_std,
            'seed': seed,
        }
        torch.save(model.state_dict(), out_dir / f"{name}.pkl")
        (out_dir / f"{name}.json").write_text(json.dumps(config | metrics, indent=2))
        print(f"[{args.study}] {name}: train={metrics['train_loss']:.5f} test={metrics['test_loss']:.5f} ({metrics['duration_sec']:.0f}s)")

    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit one grid point of the benchmark/GRU grid search for one study.')
    parser.add_argument('--study', required=True, choices=list(STUDIES))
    parser.add_argument('--kind', required=True, choices=('gru', 'benchmark'))
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--scheduler', type=int, choices=(0, 1), required=True)
    parser.add_argument('--hidden_size', type=int, default=None, help='GRU only')
    parser.add_argument('--dropout', type=float, default=None, help='GRU only')
    parser.add_argument('--init_std', type=float, default=0.5, help='benchmark only: std of the random start around the default init')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.kind == 'gru' and (args.hidden_size is None or args.dropout is None):
        parser.error('--kind gru requires --hidden_size and --dropout')
    if args.kind == 'benchmark':
        args.hidden_size = args.dropout = None
    else:
        args.init_std = None
    return args


if __name__ == '__main__':
    main()
