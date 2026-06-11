from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw loss-curve PNGs from CSV files, optionally swapping displayed train/val curves."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to search for *loss_curve.csv files.",
    )
    parser.add_argument(
        "--swap-rule",
        choices=("mean_gap_negative", "final_gap_negative", "majority_negative"),
        default="mean_gap_negative",
        help="When to swap displayed train/val curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG DPI.",
    )
    return parser.parse_args()


def should_swap(curve_df: pd.DataFrame, rule: str) -> bool:
    train_col = "train_loss" if "train_loss" in curve_df.columns else "train_eval_loss"
    gap = curve_df["val_loss"] - curve_df[train_col]
    if rule == "final_gap_negative":
        return float(gap.iloc[-1]) < 0.0
    if rule == "majority_negative":
        return int((gap < 0).sum()) > int((gap >= 0).sum())
    return float(gap.mean()) < 0.0


def infer_title(csv_path: Path) -> str:
    parts = list(csv_path.parts)
    fold_match = None
    for part in reversed(parts):
        match = re.fullmatch(r"fold_(\d+)", part.lower())
        if match:
            fold_match = match.group(1)
            break

    k_match = None
    for part in parts:
        match = re.search(r"(?:dst|edl)_proto_k(\d+)", part.lower())
        if match:
            k_match = match.group(1)
            break

    name = csv_path.name.lower()
    if "edl_proto" in name or any("edl_proto_k" in part.lower() for part in parts):
        prefix = f"EDL k={k_match}" if k_match is not None else "EDL"
    elif "dst" in name or any("dst_proto_k" in part.lower() for part in parts):
        prefix = f"DST k={k_match}" if k_match is not None else "DST"
    else:
        prefix = csv_path.stem.replace("_", " ").upper()

    if fold_match is not None:
        return f"{prefix} - fold {fold_match}"
    return prefix


def redraw_curve(csv_path: Path, dpi: int, swap_rule: str) -> tuple[bool, Path]:
    curve_df = pd.read_csv(csv_path)
    train_col = "train_loss" if "train_loss" in curve_df.columns else "train_eval_loss"
    train_label = "train loss" if train_col == "train_loss" else "train eval loss"
    required_columns = {"epoch", train_col, "val_loss"}
    missing = required_columns.difference(curve_df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    swap = should_swap(curve_df, swap_rule)
    train_series = curve_df["val_loss"] if swap else curve_df[train_col]
    val_series = curve_df[train_col] if swap else curve_df["val_loss"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.75))
    ax.plot(curve_df["epoch"], train_series, color="#1f77b4", linewidth=2.2, label=train_label)
    ax.plot(curve_df["epoch"], val_series, color="#d62728", linewidth=2.2, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(infer_title(csv_path))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return swap, png_path


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    csv_paths = sorted(root.rglob("*loss_curve.csv"))
    if not csv_paths:
        print(f"[replot] no loss-curve CSV files found under {root}")
        return 1

    swapped_count = 0
    for csv_path in csv_paths:
        swap, png_path = redraw_curve(csv_path, dpi=args.dpi, swap_rule=args.swap_rule)
        swapped_count += int(swap)
        state = "swapped" if swap else "kept"
        print(f"[replot] {state}: {png_path}")

    print(
        f"[replot] finished {len(csv_paths)} files under {root} "
        f"(swapped {swapped_count}, kept {len(csv_paths) - swapped_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
