"""Run reproducible reviewer-requested Prototype-DST ablations from bag embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable


@dataclass(frozen=True)
class Variant:
    name: str
    k: int = 10
    attract: float = 0.1
    separation: float = 0.1
    diversity: float = 0.01
    gamma_sep: float = 1.0
    gamma_div: float = 1.0


STANDARD9 = (
    Variant("full_k10"),
    Variant("k5", k=5),
    Variant("k20", k=20),
    Variant("no_attraction", attract=0.0),
    Variant("no_separation", separation=0.0),
    Variant("no_diversity", diversity=0.0),
    Variant("no_regularization", attract=0.0, separation=0.0, diversity=0.0),
    Variant("margin_0p5", gamma_sep=0.5, gamma_div=0.5),
    Variant("margin_1p5", gamma_sep=1.5, gamma_div=1.5),
)


REVIEWER17 = (
    Variant("full_k10"),
    Variant("k5", k=5),
    Variant("k20", k=20),
    Variant("no_attraction", attract=0.0),
    Variant("lambda_att_0p01", attract=0.01),
    Variant("lambda_att_1", attract=1.0),
    Variant("no_separation", separation=0.0),
    Variant("lambda_sep_0p01", separation=0.01),
    Variant("lambda_sep_1", separation=1.0),
    Variant("no_diversity", diversity=0.0),
    Variant("lambda_div_0p001", diversity=0.001),
    Variant("lambda_div_0p1", diversity=0.1),
    Variant("gamma_sep_0p5", gamma_sep=0.5),
    Variant("gamma_sep_1p5", gamma_sep=1.5),
    Variant("gamma_div_0p5", gamma_div=0.5),
    Variant("gamma_div_1p5", gamma_div=1.5),
    Variant("no_regularization", attract=0.0, separation=0.0, diversity=0.0),
)


REVIEWER9 = (
    Variant("full_k10"),
    Variant("k5", k=5),
    Variant("k20", k=20),
    Variant("no_attraction", attract=0.0),
    Variant("no_separation", separation=0.0),
    Variant("no_diversity", diversity=0.0),
    Variant("no_regularization", attract=0.0, separation=0.0, diversity=0.0),
    Variant("gamma_sep_0p5", gamma_sep=0.5),
    Variant("gamma_div_0p5", gamma_div=0.5),
)


PRESETS = {
    "reviewer9": REVIEWER9,
    "reviewer17": REVIEWER17,
    "standard9": STANDARD9,
}


VARIANT_FAMILIES = {
    "component": ("no_attraction", "no_separation", "no_diversity", "no_regularization"),
    "prototype_count": ("k5", "k20"),
    "lambda_attraction": ("no_attraction", "lambda_att_0p01", "lambda_att_1"),
    "lambda_separation": ("no_separation", "lambda_sep_0p01", "lambda_sep_1"),
    "lambda_diversity": ("no_diversity", "lambda_div_0p001", "lambda_div_0p1"),
    "gamma_separation": ("gamma_sep_0p5", "gamma_sep_1p5"),
    "gamma_diversity": ("gamma_div_0p5", "gamma_div_1p5"),
    "legacy_shared_margin": ("margin_0p5", "margin_1p5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-ready embedding-based Prototype-DST ablations."
    )
    parser.add_argument("--preset", default="reviewer9", choices=sorted(PRESETS))
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default="./origin_bag_embeddings_cancer_finetuned",
        type=Path,
    )
    parser.add_argument("--data-dir", default="/home/dhao4/workspace/hjj_workspace/data")
    parser.add_argument("--csv-file", default="data.csv")
    parser.add_argument("--dataset", default="ViNDr")
    parser.add_argument("--label", default="cancer")
    parser.add_argument("--train-cohorts", default="1-8")
    parser.add_argument("--test-cohorts", default="9-10")
    parser.add_argument(
        "--gpu-id",
        default="2",
        help="One physical GPU ID; retained as the single-GPU shorthand.",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=None,
        help=(
            "Physical GPU IDs to schedule concurrently, for example --gpu-ids 0 1 2. "
            "Each training subprocess is restricted to its assigned GPU."
        ),
    )
    parser.add_argument("--n-folds", default=5, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--allow-patient-overlap",
        action="store_true",
        help=(
            "Exploratory-only override for known patient overlap across train/test cohorts. "
            "The run manifest records this as invalid for leakage-free paper results."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--trainer",
        default=Path(__file__).resolve().with_name("edl_proto_train.py"),
        type=Path,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="New output root. Defaults to ablation_runs/embedding_dst_<preset>_<timestamp>.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and print commands only.")
    parser.add_argument(
        "--skip-cache-validation",
        action="store_true",
        help="Only allowed with --dry-run; useful when preparing commands on a machine without the cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing run root and skip variants already marked completed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later variants after a training failure.",
    )
    parser.add_argument(
        "--extra-trainer-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended verbatim to every edl_proto_train.py command.",
    )
    return parser.parse_args()


def choose_variants(spec: str, preset: str = "reviewer9") -> list[Variant]:
    available = PRESETS[preset]
    by_name = {variant.name: variant for variant in available}
    if spec.strip().lower() == "all":
        return list(available)
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = sorted(set(names).difference(by_name))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(by_name)}")
    if len(names) != len(set(names)):
        raise ValueError("--variants contains duplicate names.")
    return [by_name[name] for name in names]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(cache_dir: Path) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    cache_dir = cache_dir.expanduser().resolve()
    required = {
        "manifest": cache_dir / "manifest.json",
        "metadata": cache_dir / "metadata.csv",
        "embeddings": cache_dir / "embeddings.npy",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Embedding cache is incomplete; missing: " + ", ".join(missing))

    with required["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("embedding_level") != "bag_origin":
        raise ValueError(
            "The embedding route requires manifest embedding_level='bag_origin', "
            f"got {manifest.get('embedding_level')!r}."
        )

    metadata = pd.read_csv(required["metadata"], dtype={"patient_id": str, "image_id": str})
    expected_columns = {
        "patient_id",
        "image_id",
        "embedding_start",
        "embedding_end",
        "origin_prediction_score",
    }
    missing_columns = sorted(expected_columns.difference(metadata.columns))
    if missing_columns:
        raise ValueError(f"metadata.csv is missing columns: {missing_columns}")
    if metadata.empty:
        raise ValueError("metadata.csv contains no samples.")
    if metadata[["patient_id", "image_id"]].isna().any().any():
        raise ValueError("metadata.csv contains null patient_id/image_id values.")
    duplicate_count = int(metadata.duplicated(["patient_id", "image_id"]).sum())
    if duplicate_count:
        raise ValueError(f"metadata.csv contains {duplicate_count} duplicate patient/image keys.")

    embeddings = np.load(required["embeddings"], mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings.npy must be 2D, got shape={embeddings.shape}.")
    starts = metadata["embedding_start"].astype(int).to_numpy()
    ends = metadata["embedding_end"].astype(int).to_numpy()
    if not np.all(ends - starts == 1):
        raise ValueError("bag_origin cache must have exactly one embedding row per image.")
    if starts.min() < 0 or ends.max() > embeddings.shape[0]:
        raise ValueError("metadata embedding ranges fall outside embeddings.npy.")
    if len(metadata) != embeddings.shape[0] or len(np.unique(starts)) != len(starts):
        raise ValueError("metadata rows and embedding rows are not a one-to-one mapping.")
    scores = pd.to_numeric(metadata["origin_prediction_score"], errors="raise")
    if not np.isfinite(scores).all() or not scores.between(0.0, 1.0).all():
        raise ValueError("metadata origin_prediction_score must be finite and lie in [0, 1].")

    return {
        "path": str(cache_dir),
        "embedding_level": "bag_origin",
        "samples": int(len(metadata)),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "dtype": str(embeddings.dtype),
        "manifest_sha256": file_sha256(required["manifest"]),
        "metadata_sha256": file_sha256(required["metadata"]),
    }


def build_command(
    args: argparse.Namespace,
    variant: Variant,
    variant_root: Path,
    trainer_gpu_id: str = "0",
) -> list[str]:
    return [
        str(args.python),
        str(args.trainer.resolve()),
        "--gpu_id", str(trainer_gpu_id),
        "--output_dir", str(variant_root.resolve()),
        "--data_dir", str(args.data_dir),
        "--csv_file", str(args.csv_file),
        "--embedding_cache_dir", str(args.embedding_cache_dir.expanduser().resolve()),
        "--feature_extraction", "bag_embedding",
        "--dataset", str(args.dataset),
        "--label", str(args.label),
        "--n_folds", str(args.n_folds),
        "--start-fold", "0",
        "--train-cohorts", str(args.train_cohorts),
        "--test-cohorts", str(args.test_cohorts),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--weighted-BCE", "y",
        "--early_stop_patience", "0",
        "--apex", "y",
        "--edl_proto_k", str(variant.k),
        "--edl_proto_topk", str(min(3, variant.k)),
        "--edl_proto_init", "fold_best_scores",
        "--edl_proto_normalize", "y",
        "--edl_proto_balance_classes", "y",
        "--edl_proto_allow_patient_overlap", "y" if getattr(args, "allow_patient_overlap", False) else "n",
        "--edl_proto_attract_weight", str(variant.attract),
        "--edl_proto_separation_weight", str(variant.separation),
        "--edl_proto_diversity_weight", str(variant.diversity),
        "--edl_proto_gamma_sep", str(variant.gamma_sep),
        "--edl_proto_gamma_div", str(variant.gamma_div),
        *[str(value) for value in args.extra_trainer_args],
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_status_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "variant",
        "gpu_id",
        "state",
        "attempt",
        "started_at",
        "finished_at",
        "returncode",
        "output_dir",
        "log",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stream_command(
    command: list[str],
    log_path: Path,
    visible_gpu_id: str,
    prefix: str = "",
) -> int:
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_id)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"{prefix}{line}", end="")
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def main() -> int:
    args = parse_args()
    if args.skip_cache_validation and not args.dry_run:
        raise ValueError("--skip-cache-validation is only allowed with --dry-run.")
    if args.n_folds != 5:
        raise ValueError("Paper ablation presets are defined for exactly five folds.")
    if not args.trainer.is_file():
        raise FileNotFoundError(f"Trainer not found: {args.trainer}")
    gpu_ids = [str(gpu_id) for gpu_id in (args.gpu_ids or [args.gpu_id])]
    if not gpu_ids or any(not gpu_id.strip() for gpu_id in gpu_ids):
        raise ValueError("At least one non-empty GPU ID is required.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicates: {gpu_ids}")

    variants = choose_variants(args.variants, args.preset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else (
            Path(__file__).resolve().parent
            / "ablation_runs"
            / f"embedding_dst_{args.preset}_{timestamp}"
        )
    )
    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Run root is not empty: {run_root}. Use --resume or choose a new root.")
    run_root.mkdir(parents=True, exist_ok=True)

    cache_info = None if args.skip_cache_validation else validate_cache(args.embedding_cache_dir)
    attempt_name = f"attempt_{timestamp}"
    attempt_roots = {
        variant.name: run_root / variant.name / attempt_name for variant in variants
    }
    commands = {
        variant.name: build_command(args, variant, attempt_roots[variant.name])
        for variant in variants
    }
    manifest = {
        "schema_version": 2,
        "preset": args.preset,
        "invocation_id": timestamp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "trainer": str(args.trainer.resolve()),
        "gpu_ids": gpu_ids,
        "gpu_mapping": (
            "Each trainer sees its assigned physical GPU as CUDA device 0 via "
            "CUDA_VISIBLE_DEVICES."
        ),
        "cache": cache_info,
        "fixed_settings": {
            "feature_extraction": "bag_embedding",
            "prototype_init": "fold_best_scores",
            "normalize": True,
            "balance_classes": True,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "weighted_BCE": True,
            "patient_overlap_override": bool(getattr(args, "allow_patient_overlap", False)),
            "train_cohorts": args.train_cohorts,
            "test_cohorts": args.test_cohorts,
        },
        "variants": [asdict(variant) for variant in variants],
        "analysis_families": {
            family: [name for name in names if name in commands]
            for family, names in VARIANT_FAMILIES.items()
        },
        "commands": commands,
    }
    write_json(run_root / "run_manifest.json", manifest)
    write_json(run_root / "run_manifests" / f"run_manifest_{timestamp}.json", manifest)

    print(f"Run root: {run_root}")
    print(f"GPU scheduler: {', '.join(gpu_ids)} ({len(gpu_ids)} concurrent worker(s))")
    for variant in variants:
        print(f"[{variant.name}] {shlex.join(commands[variant.name])}")
    if args.dry_run:
        print("Dry run complete; no training processes were started.")
        return 0

    status_path = run_root / "run_status.json"
    existing_status = {}
    if args.resume and status_path.is_file():
        existing_status = json.loads(status_path.read_text(encoding="utf-8"))
    status: dict[str, dict[str, object]] = dict(existing_status)

    pending_variants = []
    for variant in variants:
        previous = status.get(variant.name, {})
        if args.resume and previous.get("state") == "completed":
            print(f"[{variant.name}] already completed; skipping.")
            continue
        pending_variants.append(variant)

    status_lock = Lock()

    def run_variant(variant: Variant, physical_gpu_id: str) -> tuple[str, int]:
        attempt_root = attempt_roots[variant.name]
        attempt_root.mkdir(parents=True, exist_ok=True)
        log_path = attempt_root / "training.log"
        command = build_command(args, variant, attempt_root, trainer_gpu_id="0")
        with status_lock:
            status[variant.name] = {
                "variant": variant.name,
                "gpu_id": physical_gpu_id,
                "state": "running",
                "attempt": attempt_name,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": "",
                "returncode": "",
                "output_dir": str(attempt_root),
                "log": str(log_path),
            }
            write_json(status_path, status)
            write_status_csv(run_root / "run_status.csv", status.values())

        print(f"\n===== Starting {variant.name} on physical GPU {physical_gpu_id} =====")
        returncode = stream_command(
            command,
            log_path,
            visible_gpu_id=physical_gpu_id,
            prefix=f"[{variant.name} | gpu {physical_gpu_id}] ",
        )
        with status_lock:
            status[variant.name]["state"] = "completed" if returncode == 0 else "failed"
            status[variant.name]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            status[variant.name]["returncode"] = returncode
            write_json(status_path, status)
            write_status_csv(run_root / "run_status.csv", status.values())
        return variant.name, returncode

    pending_index = 0
    active = {}
    failed_returncode = None
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        for physical_gpu_id in gpu_ids:
            if pending_index >= len(pending_variants):
                break
            variant = pending_variants[pending_index]
            pending_index += 1
            active[executor.submit(run_variant, variant, physical_gpu_id)] = physical_gpu_id

        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                physical_gpu_id = active.pop(future)
                variant_name, returncode = future.result()
                if returncode != 0 and failed_returncode is None:
                    failed_returncode = returncode
                    if not args.continue_on_error:
                        print(
                            f"[{variant_name}] failed with return code {returncode}; "
                            "no further variants will be scheduled.",
                            file=sys.stderr,
                        )
                if pending_index < len(pending_variants) and (
                    args.continue_on_error or failed_returncode is None
                ):
                    variant = pending_variants[pending_index]
                    pending_index += 1
                    active[executor.submit(run_variant, variant, physical_gpu_id)] = physical_gpu_id

    if failed_returncode is not None and not args.continue_on_error:
        return failed_returncode

    failures = [name for name, row in status.items() if row.get("state") == "failed"]
    if failures:
        print(f"Completed with failed variants: {failures}", file=sys.stderr)
        return 1
    print("All selected Prototype-DST ablations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
