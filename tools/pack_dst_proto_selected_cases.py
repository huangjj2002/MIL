#!/usr/bin/env python3
"""Pack selected DST-Prototype cases and their top-1 prototype images.

This script is intended to run on the server where the original PNG images live.
It reads the organized CSV produced from prediction results, copies selected
case images and their top-1 prototype images into short-name folders, and writes
a tar.gz archive.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="selected_TP_TN_low_uncertainty_top1_*.csv from the organized folder.",
    )
    parser.add_argument(
        "--img-root",
        default=Path("/home/dhao4/workspace/hjj_workspace/data/images_png"),
        type=Path,
        help="Directory containing original PNG images on the server.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("dst_proto_selected_cases_all_splits_no_proto_2026-06-14"),
        type=Path,
        help="Output folder to create before archiving.",
    )
    parser.add_argument(
        "--archive",
        default=Path("dst_proto_selected_cases_all_splits_no_proto_2026-06-14.tar.gz"),
        type=Path,
        help="Output tar.gz path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output folder/archive before packing.",
    )
    return parser.parse_args()


def copy_if_present(src: Path, dst: Path, missing: list[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing.append(str(src))


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.img_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.img_root}")

    if args.overwrite:
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        if args.archive.exists():
            args.archive.unlink()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied_cases = 0
    copied_prototypes = 0

    with args.csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "selection_group",
            "sample_short_name",
            "image_id",
            "top1_prototype_pack_name",
            "top1_prototype_source_image_id",
        }
        missing_cols = required.difference(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

        for row in reader:
            group = row["selection_group"]
            case_src = args.img_root / row["image_id"]
            case_dst = args.out_dir / group / "cases" / row["sample_short_name"]
            proto_src = args.img_root / row["top1_prototype_source_image_id"]
            proto_dst = (
                args.out_dir
                / group
                / "top1_prototypes"
                / row["top1_prototype_pack_name"]
            )

            before = len(missing)
            copy_if_present(case_src, case_dst, missing)
            copied_cases += int(len(missing) == before)

            before = len(missing)
            copy_if_present(proto_src, proto_dst, missing)
            copied_prototypes += int(len(missing) == before)

    readme = args.out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Scope: train/val/test predictions, excluding prototype source images as selected cases.",
                "TP_low_uncertainty/cases: true-positive cases renamed TP_01.png ... TP_20.png.",
                "TN_low_uncertainty/cases: true-negative cases renamed TN_01.png ... TN_20.png.",
                "*/top1_prototypes: each selected case's predicted-class top1 prototype image.",
                "Uncertainty function used in CSV: uncertainty = mass_omega = m(Omega).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if missing:
        (args.out_dir / "missing_files.txt").write_text(
            "\n".join(missing) + "\n",
            encoding="utf-8",
        )

    with tarfile.open(args.archive, "w:gz") as tar:
        tar.add(args.out_dir, arcname=args.out_dir.name)

    print(f"copied cases: {copied_cases}")
    print(f"copied top1 prototypes: {copied_prototypes}")
    print(f"missing files: {len(missing)}")
    print(f"output dir: {args.out_dir}")
    print(f"archive: {args.archive}")


if __name__ == "__main__":
    main()
