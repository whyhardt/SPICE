import torch
from spice import SpiceDataset, dataset_to_csv


def generate_repeated(gen_fn, n_repeats=100, save_csv=None, **kwargs):
    """Generate behavior n_repeats times and concatenate for stable Monte Carlo estimates.

    Calls ``gen_fn(**kwargs)`` repeatedly and concatenates the resulting
    SpiceDatasets along the session dimension. This produces
    ``n_repeats * n_sessions`` sessions with the same participant/block
    metadata, allowing downstream analyses to average over repeats for
    stable per-participant statistics.

    Args:
        gen_fn: Generation function (e.g. ``generate_behavior`` from a
            study's benchmarking module).  Must return a SpiceDataset.
        n_repeats: Number of independent stochastic repetitions.
        save_csv: Optional path to save concatenated dataset as CSV.
        **kwargs: Forwarded to ``gen_fn``.

    Returns:
        SpiceDataset with ``n_repeats * n_sessions`` sessions.
    """
    all_xs, all_ys = [], []
    for r in range(n_repeats):
        ds = gen_fn(**kwargs)
        xs_r = ds.xs.clone()
        # Offset block IDs so each repeat produces unique (participant, block)
        # sessions when saved to CSV and reloaded via get_dataset.
        if r > 0:
            max_block = int(ds.xs[:, 0, 0, -3].max().item()) + 1
            xs_r[:, :, :, -3] = xs_r[:, :, :, -3] + r * max_block
        all_xs.append(xs_r)
        all_ys.append(ds.ys)
    ds_cat = SpiceDataset(
        torch.cat(all_xs, dim=0),
        torch.cat(all_ys, dim=0),
        n_reward_features=ds.n_reward_features,
        continuous_action=getattr(ds, 'continuous_action', False),
    )
    if save_csv is not None:
        dataset_to_csv(ds_cat, save_csv)
    return ds_cat
