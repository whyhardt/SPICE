"""
Run-table generator for Slurm array sweeps.
- generates a JSONL run table where each line represents one run 
(study x model x hyperparams x replicate).

The run table is deliberately explicit about:
- dataset loading (entrypoint + kwargs, or csv_to_dataset kwargs)
- model construction (module/class + kwargs and which args are injected
  from the loaded dataset, e.g. n_actions)
- training hyperparameters 
    - For GRU benchmark: 2x2x2x3x4=96: lr (1e-2, 1e-3) x epochs (1000, 10000) x scheduler (True, False) x hidden_size (8, 16, 32) x dropout (0, 0.1, 0.25, 0.5)
    - For other BMs:     2x2x2=8: lr (1e-2, 1e-3) x epochs (1000, 10000) x scheduler (True, False)
- deterministic seeding (base_seed + stable hash)

Note: 
A separate runner script will read this run table (one row at a time) and do the actual training.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------
# Search space / constants
# ---------------------------

# Base path for all study folders
WEINHARDT_DIR = Path(__file__).resolve().parents[1]
STUDIES_DIR = WEINHARDT_DIR / "studies"
DATA_DIR = WEINHARDT_DIR / "data"  # Shared data dir for now
OUTPUT_DIR = WEINHARDT_DIR / "slurm_training" / "json_outputs"

LRS: Tuple[float, ...] = (1e-2, 1e-3)
EPOCHS: Tuple[int, ...] = (1000, 10000)
SCHEDULERS: Tuple[bool, ...] = (True, False)
GRU_HIDDEN_SIZES: Tuple[int, ...] = (8, 16, 32)
GRU_DROPOUTS: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)


@dataclass(frozen=True)
class StudySpec:
	key: str
	workdir: str
	# Dataset loading strategy:
	# - If dataset_entrypoint is set, runner should call it and expect
	#   `(dataset_train, dataset_test, info_dict)`.
	# - Else (Huang/Braun don't use get_dataset), runner should call `spice.csv_to_dataset(**csv_to_dataset_kwargs)`
	#   and then split via split_data_along_blockdim(test_blocks=...) (this is old:`split_data_along_sessiondim(test_sessions=...)`.

	dataset_entrypoint: Optional[str]
	path_data: str
	test_blocks: Sequence[int]
	csv_to_dataset_kwargs: Optional[Dict[str, Any]] = None
	normalize_rewards: bool = True

	# Benchmark model (study-specific hand-coded model)
	benchmark_class: str = ""
	benchmark_init_kwargs: Optional[Dict[str, Any]] = None


STUDIES: Tuple[StudySpec, ...] = (
	# NOTE: Braun/Huang benchmarking modules currently define only the model
	# class. Dataset loading is taken from their notebooks.
	StudySpec(
		key="braun2018",
		workdir=str(STUDIES_DIR / "braun2018"),
		dataset_entrypoint=None,
		path_data=str(DATA_DIR / "braun2018.csv"),
		test_blocks=(3, 6, 9),
		csv_to_dataset_kwargs={
			"df_participant_id": "subject",
			"df_choice": "transcode",
			"df_feedback": None,
			"df_block": "block",
			"additional_inputs": ("difference", "current", "other"),
			"timeshift_additional_inputs": (-1, -1, -1),
		},
		normalize_rewards=False,
		benchmark_class="weinhardt2026.studies.braun2018.benchmarking_braun2018:ExpectedValueControl",
		benchmark_init_kwargs={"batch_first": True},
	),
	StudySpec(
		key="bruckner2025",
		workdir=str(STUDIES_DIR / "bruckner2025"),
		dataset_entrypoint="weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025:get_dataset",
		path_data=str(DATA_DIR / "bruckner2025.csv"),
		test_blocks=(1, 2),
		benchmark_class="weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025:RationalResourceModel",
		benchmark_init_kwargs={},
	),
	StudySpec(
		key="bustamante2023",
		workdir=str(STUDIES_DIR / "bustamante2023"),
		dataset_entrypoint="weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023:get_dataset",
		path_data=str(DATA_DIR / "bustamante2023.csv"),
		test_blocks=(3, 6),
		benchmark_class="weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023:MarginalValueTheoremModel",
		benchmark_init_kwargs={"batch_first": True},
	),
	StudySpec(
		key="castro2025",
		workdir=str(STUDIES_DIR / "castro2025"),
		dataset_entrypoint="weinhardt2026.studies.castro2025.benchmarking_castro2025:get_dataset",
		path_data= str(DATA_DIR / "eckstein2024.csv"),
		test_blocks=(2,),
		benchmark_class="weinhardt2026.studies.castro2025.benchmarking_castro2025:Castro2025Model",
		benchmark_init_kwargs={"batch_first": True},
	),
	StudySpec(
		key="dezfouli2019",
		workdir=str(STUDIES_DIR / "dezfouli2019"),
		dataset_entrypoint="weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019:get_dataset",
		path_data=str(DATA_DIR / "dezfouli2019.csv"),
		test_blocks=(3, 6, 9),
		benchmark_class="weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019:GQLModel",
		benchmark_init_kwargs={"batch_first": True},
	),
	StudySpec(
		key="ganesh2024a",
		workdir=str(STUDIES_DIR / "ganesh2024a"),
		dataset_entrypoint="weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a:get_dataset",
		path_data=str(DATA_DIR / "ganesh2024a_choice.csv"),
		test_blocks=(3, 6, 9),
		benchmark_class="weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a:BayesianModel",
		benchmark_init_kwargs={"batch_first": True},
	),
	StudySpec(
		key="huang2026",
		workdir=str(STUDIES_DIR / "huang2026"),
		dataset_entrypoint=None,
		path_data=str(DATA_DIR / "huang2025.csv"),
		test_blocks=tuple(range(10, 40, 3)),
		csv_to_dataset_kwargs={
			"df_participant_id": "subject_ID",
			"df_block": "currentRound",
			"df_choice": "hover_tile_index",
			"df_feedback": None,
			"additional_inputs": ["partner_tile_index", "time_point"],
		},
		normalize_rewards=False,
		benchmark_class="weinhardt2026.studies.huang2026.benchmarking_huang2026:InformationForagingModel",
		benchmark_init_kwargs={},
	),
	StudySpec(
		key="kolff2025",
		workdir=str(STUDIES_DIR / "kolff2025"),
		dataset_entrypoint="weinhardt2026.studies.kolff2025.benchmarking_kolff2025:get_dataset",
		path_data=str(DATA_DIR / "kolff2025.csv"),
		test_blocks=(1,),
		benchmark_class="weinhardt2026.studies.kolff2025.benchmarking_kolff2025:ConditionalFrequencyModel",
		benchmark_init_kwargs={},
	),
	StudySpec(
		key="weber2024",
		workdir=str(STUDIES_DIR / "weber2024"),
		dataset_entrypoint="weinhardt2026.studies.weber2024.benchmarking_weber2024:get_dataset",
		path_data=str(DATA_DIR / "weber2024.csv"),
		test_blocks=(2, 5, 9, 14),
		benchmark_class="weinhardt2026.studies.weber2024.benchmarking_weber2024:ChangePointModel",
		benchmark_init_kwargs={"batch_first": True},
	),
)

STUDY_KEYS: Tuple[str, ...] = tuple(study.key for study in STUDIES)

# deterministic seed generation from a text key that's stable across Python versions
# pyhtons build in hash is randomised per run, so can't rely to reproduction
def _stable_u32(text: str) -> int:
	"""Stable 32-bit-ish hash for seeds/ids (independent of PYTHONHASHSEED)."""
	h = hashlib.sha1(text.encode("utf-8")).hexdigest() # Take first 8 chars (32 bits) for the seed
	return int(h[:8], 16)

# generate stable, short config IDs based on hyperparams 
# group results of all 20 replicates of the same config under the same ID, without comparing full dicts
def _config_id(prefix: str, payload: Dict[str, Any]) -> str:
	blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
	return f"{prefix}_{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:10]}"

# generate all combinations of hyperparams from dict like {"lr": [1e-2, 1e-3], "epochs": [1000, 10000]} -> [{"lr": 1e-2, "epochs": 1000}, {"lr": 1e-2, "epochs": 10000}, ...]
# use itertools.product but keep track of keys and return dicts for readability
def _iter_product(mapping: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
	keys = list(mapping.keys())
	for values in itertools.product(*(mapping[k] for k in keys)):
		yield dict(zip(keys, values))

# Main function for run table generations - one entry per run instance (study x model x hyperparams x replicate)
def build_runs(
	*,
	n_replicates: int,
	base_seed: int,
	include_gru: bool,
	benchmark_keys: Set[str],
	gru_keys: Set[str],
) -> List[Dict[str, Any]]:
	runs: List[Dict[str, Any]] = []

	for study in STUDIES:
		# -----------------
		# Benchmarks
		# -----------------
		if study.key in benchmark_keys:
			for hp in _iter_product({"lr": LRS, "epochs": EPOCHS, "scheduler": SCHEDULERS}):
				cfg_id = _config_id(f"bench_{study.key}", {"benchmark_class": study.benchmark_class, **hp})
				for rep in range(n_replicates):
					key = f"{study.key}|benchmark|{cfg_id}|rep={rep}"
					seed = int(base_seed) + _stable_u32(key)
					runs.append(
						{
							"run_key": key,
							"seed": seed,
							"study": {
								"key": study.key,
								"workdir": study.workdir,
							},
							"dataset": {
								"path_data": study.path_data,
								"test_blocks": list(study.test_blocks),
								"normalize_rewards": bool(study.normalize_rewards),
								"loader": (
									{
										"type": "entrypoint",
										"call": study.dataset_entrypoint,
										"kwargs": {
											"path_data": study.path_data,
											"test_blocks": tuple(study.test_blocks),
																					},
									}
									if study.dataset_entrypoint
									else {
										"type": "csv_to_dataset",
										"call": "spice.utils.convert_dataset:csv_to_dataset",
										"kwargs": {"file": study.path_data, **(study.csv_to_dataset_kwargs or {})},
										"split": {
											"call": "spice:split_data_along_blockdim",
											"kwargs": {"test_blocks": list(study.test_blocks)},
										},
									}
								),
							},
							"model": {
								"kind": "benchmark",
								"cfg_id": cfg_id,
								"replicate": rep,
								"class": study.benchmark_class,
								"init_kwargs": study.benchmark_init_kwargs or {},
								# Runner should read these off the loaded dataset and inject them.
								"inject_from_dataset": {
									"n_actions": "dataset.n_actions",
									"n_participants": "dataset.n_participants",
								},
							},
							"training": {
								"lr": float(hp["lr"]),
								"epochs": int(hp["epochs"]),
								"optimizer": "adam",
								# Keep defaults in `training()` unless runner overrides.
								"batch_size": None,
								"scheduler": bool(hp["scheduler"]),
							},
						}
					)

		# -----------------
		# GRU baseline
		# -----------------
		if include_gru and study.key in gru_keys:
			for hp in _iter_product(
				{
					"lr": LRS,
					"epochs": EPOCHS,
					"hidden_size": GRU_HIDDEN_SIZES,
					"dropout": GRU_DROPOUTS,
					"scheduler": SCHEDULERS,
				}
			):
				cfg_id = _config_id(f"gru_{study.key}", hp)
				for rep in range(n_replicates):
					key = f"{study.key}|gru|{cfg_id}|rep={rep}"
					seed = int(base_seed) + _stable_u32(key)
					runs.append(
						{
							"run_key": key,
							"seed": seed,
							"study": {
								"key": study.key,
								"workdir": study.workdir,
							},
							"dataset": {
								"path_data": study.path_data,
								"test_blocks": list(study.test_blocks),
								"normalize_rewards": bool(study.normalize_rewards),
								"loader": (
									{
										"type": "entrypoint",
										"call": study.dataset_entrypoint,
										"kwargs": {
											"path_data": study.path_data,
											"test_blocks": tuple(study.test_blocks),
																					},
									}
									if study.dataset_entrypoint
									else {
										"type": "csv_to_dataset",
										"call": "spice.utils.convert_dataset:csv_to_dataset",
										"kwargs": {"file": study.path_data, **(study.csv_to_dataset_kwargs or {})},
										"split": {
											"call": "spice:split_data_along_blockdim",
											"kwargs": {"test_blocks": list(study.test_blocks)},
										},
									}
								),
							},
							"model": {
								"kind": "gru",
								"cfg_id": cfg_id,
								"replicate": rep,
								"class": "weinhardt2026.utils.benchmarking_gru:GRUModel",
								"init_kwargs": {
									"hidden_size": int(hp["hidden_size"]),
									"dropout": float(hp["dropout"]),
								},
								"inject_from_dataset": {
									"n_actions": "dataset.n_actions",
									"n_reward_features": "dataset.n_reward_features",
									"additional_inputs": "dataset.n_additional_inputs",
									"n_participants": "dataset.n_participants",
									"n_experiments": "dataset.n_experiments",
								},
							},
							"training": {
								"lr": float(hp["lr"]),
								"epochs": int(hp["epochs"]),
								"optimizer": "adam",
								"batch_size": None,
								"scheduler": bool(hp["scheduler"]),
							},
						}
					)

	# Stable ordering for reproducible Slurm indexing.
	runs.sort(key=lambda r: (r["study"]["key"], r["model"]["kind"], r["model"]["cfg_id"], r["model"]["replicate"]))
	for idx, row in enumerate(runs):
		row["run_index"] = idx
	return runs


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_study_list(value: str) -> Set[str]:
	items = [item.strip() for item in value.split(",") if item.strip()]
	if not items:
		raise ValueError("Study list must include at least one token.")
	if "all" in items:
		return set(STUDY_KEYS)
	unknown = [item for item in items if item not in STUDY_KEYS]
	if unknown:
		raise ValueError(f"Unknown study keys: {unknown}. Allowed: {sorted(STUDY_KEYS)}")
	return set(items)


def _parse_only(value: str) -> Tuple[bool, Set[str]]:
	"""Parse --only selection into (include_gru, benchmark_keys)."""
	items = [item.strip() for item in value.split(",") if item.strip()]
	if not items:
		raise ValueError("--only must include at least one token.")

	allowed = {"all", "gru", "all_bms", *STUDY_KEYS}
	unknown = [item for item in items if item not in allowed]
	if unknown:
		raise ValueError(f"Unknown --only values: {unknown}. Allowed: {sorted(allowed)}")

	include_gru = "all" in items or "gru" in items
	if "all" in items or "all_bms" in items:
		benchmark_keys = set(STUDY_KEYS)
	else:
		benchmark_keys = {item for item in items if item in STUDY_KEYS}

	return include_gru, benchmark_keys


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate JSONL run table for Slurm array sweeps.")
	p.add_argument(
		"--out",
		type=str,
		default= str(OUTPUT_DIR / "runs.jsonl"),
		help="Output JSONL file (one run per line).",
	)
	p.add_argument(
		"--n-replicates",
		type=int,
		default=20,
		help="Number of random restarts per hyperparameter configuration.",
	)
	p.add_argument(
		"--base-seed",
		type=int,
		default=1337,
		help="Base seed; actual seed is base_seed + stable_hash(run_key).",
	)
	p.add_argument(
		"--only",
		default="all",
		help=(
			"Comma-separated selection. Examples: 'all', 'gru', 'all_bms', "
			"'castro2025', 'gru,dezfouli2019', 'all_bms,gru'."
		),
	)
	p.add_argument(
		"--gru-studies",
		default="all",
		help=(
			"Comma-separated GRU study list or 'all'. Examples: 'castro2025' or 'all'."
		),
	)
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
	args = parse_args(argv)

	include_gru, benchmark_keys = _parse_only(args.only)
	gru_keys = _parse_study_list(args.gru_studies)

	runs = build_runs(
		n_replicates=int(args.n_replicates),
		base_seed=int(args.base_seed),
		include_gru=include_gru,
		benchmark_keys=benchmark_keys,
		gru_keys=gru_keys,
	)

	out_path = Path(args.out)
	write_jsonl(out_path, runs)

	# Print a small, useful summary for sanity-checking job counts.
	n_gru_cfg = len(LRS) * len(EPOCHS) * len(SCHEDULERS) * len(GRU_HIDDEN_SIZES) * len(GRU_DROPOUTS)
	n_bench_cfg = len(LRS) * len(EPOCHS) * len(SCHEDULERS)
	print(f"Wrote {len(runs)} runs -> {out_path}")
	if include_gru:
		print(
			f"GRU: {len(gru_keys)} studies x {n_gru_cfg} cfg x {args.n_replicates} reps = "
			f"{len(gru_keys) * n_gru_cfg * args.n_replicates}"
		)
	if benchmark_keys:
		print(
			f"Benchmarks: {len(benchmark_keys)} studies x {n_bench_cfg} cfg x "
			f"{args.n_replicates} reps = {len(benchmark_keys) * n_bench_cfg * args.n_replicates}"
		)


if __name__ == "__main__":
	main()

