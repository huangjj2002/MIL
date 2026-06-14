"""Export spatial attention heatmaps from an existing MIL-origin checkpoint.

This is an inference-only visualization utility. It uses the trained MIL
attention weights to produce localization-style heatmaps; it does not create
EDL/DST uncertainty maps unless the checkpoint itself was trained with such a
head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MIL attention heatmaps from a trained origin checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to fold best_model.pth.")
    parser.add_argument(
        "--source-args-yaml",
        default=None,
        type=str,
        help="Optional args.yaml from the original training run. Defaults to checkpoint/../../args.yaml if present.",
    )
    parser.add_argument("--data-dir", default="/home/dhao4/workspace/hjj_workspace/data", type=str)
    parser.add_argument("--csv-file", default="data.csv", type=str)
    parser.add_argument("--img-dir", default="images_png", type=str)
    parser.add_argument("--clip_chk_pt_path", default="./models/b2-model-best-epoch-10.tar", type=str)
    parser.add_argument("--out-dir", default="attention_heatmaps/mil_origin_fold1", type=str)
    parser.add_argument("--label", default="cancer", type=str)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--train-cohorts", default="1-8", type=str)
    parser.add_argument("--test-cohorts", default="9-10", type=str)
    parser.add_argument("--patient-id", default=None, type=str)
    parser.add_argument("--image-id", default=None, type=str)
    parser.add_argument("--only-positive", action="store_true")
    parser.add_argument("--max-images", default=12, type=int)
    parser.add_argument("--min-score", default=None, type=float)
    parser.add_argument("--top-score", action="store_true", help="Sort candidate images by model score.")
    parser.add_argument(
        "--uncertainty-csv",
        default=None,
        type=str,
        help=(
            "Optional DST/EDL prediction CSV containing patient_id, image_id, and an "
            "uncertainty column. When provided, the script also writes "
            "attention-guided uncertainty maps: normalized_attention * uncertainty."
        ),
    )
    parser.add_argument("--uncertainty-col", default="uncertainty", type=str)
    parser.add_argument(
        "--uncertainty-score-col",
        default="prediction_score",
        type=str,
        help="Optional score column in uncertainty-csv used for manifest reporting.",
    )
    parser.add_argument("--gpu-id", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--cmap", default="viridis", type=str)
    parser.add_argument("--overlay-alpha", default=0.45, type=float)
    parser.add_argument("--sigma", default=10.0, type=float, help="Gaussian blur sigma for the heatmap.")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay PNGs.")
    parser.add_argument("--save-npy", action="store_true", help="Save raw heatmaps as compressed npz files.")

    # Defaults matching the selected MIL-origin run. Values from args.yaml, when
    # available, override these defaults unless explicitly set by CLI above.
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str)
    parser.add_argument("--dataset", default="ViNDr", type=str)
    parser.add_argument("--img-size", "--img_size", dest="img_size", nargs="+", default=[1520, 912], type=int)
    parser.add_argument("--patch_size", "--patch-size", dest="patch_size", default=512, type=int)
    parser.add_argument("--overlap", nargs="*", default=[0.0], type=float)
    parser.add_argument("--scales", nargs="*", default=[16, 32, 128], type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--feat_dim", "--feat-dim", dest="feat_dim", default=352, type=int)
    parser.add_argument("--mil_type", "--mil-type", dest="mil_type", default="pyramidal_mil")
    parser.add_argument("--pooling_type", "--pooling-type", dest="pooling_type", default="gated-attention")
    parser.add_argument("--type_mil_encoder", "--type-mil-encoder", dest="type_mil_encoder", default="sab")
    parser.add_argument("--fcl_attention_dim", "--fcl-attention-dim", dest="fcl_attention_dim", default=128, type=int)
    parser.add_argument("--map_prob_func", "--map-prob-func", dest="map_prob_func", default="softmax")
    parser.add_argument("--fcl_encoder_dim", "--fcl-encoder-dim", dest="fcl_encoder_dim", default=256, type=int)
    parser.add_argument("--sab_num_heads", "--sab-num-heads", dest="sab_num_heads", default=4, type=int)
    parser.add_argument("--isab_num_heads", "--isab-num-heads", dest="isab_num_heads", default=4, type=int)
    parser.add_argument("--pma_num_heads", "--pma-num-heads", dest="pma_num_heads", default=1, type=int)
    parser.add_argument("--num_encoder_blocks", "--num-encoder-blocks", dest="num_encoder_blocks", default=2, type=int)
    parser.add_argument("--trans_num_inds", "--trans-num-inds", dest="trans_num_inds", default=20, type=int)
    parser.add_argument("--trans_layer_norm", "--trans-layer-norm", dest="trans_layer_norm", action="store_true")
    parser.add_argument("--multi_scale_model", "--multi-scale-model", dest="multi_scale_model", default="fpn")
    parser.add_argument("--fpn_dim", "--fpn-dim", dest="fpn_dim", default=256, type=int)
    parser.add_argument("--upsample_method", "--upsample-method", dest="upsample_method", default="nearest")
    parser.add_argument("--norm_fpn", "--norm-fpn", dest="norm_fpn", action="store_true")
    parser.add_argument("--deep_supervision", "--deep-supervision", dest="deep_supervision", action="store_true")
    parser.add_argument("--type_scale_aggregator", "--type-scale-aggregator", dest="type_scale_aggregator", default="gated-attention")
    parser.add_argument("--nested_model", "--nested-model", dest="nested_model", action="store_true")
    parser.add_argument("--type_region_aggregator", "--type-region-aggregator", dest="type_region_aggregator", default=None)
    parser.add_argument("--type_region_encoder", "--type-region-encoder", dest="type_region_encoder", default=None)
    parser.add_argument("--type_region_pooling", "--type-region-pooling", dest="type_region_pooling", default=None)
    parser.add_argument("--drop_classhead", "--drop-classhead", dest="drop_classhead", default=0.0, type=float)
    parser.add_argument("--drop_attention_pool", "--drop-attention-pool", dest="drop_attention_pool", default=0.25, type=float)
    parser.add_argument("--drop_mha", "--drop-mha", dest="drop_mha", default=0.0, type=float)
    parser.add_argument("--fcl_dropout", "--fcl-dropout", dest="fcl_dropout", default=0.0, type=float)
    return parser.parse_args()


def maybe_load_training_args(args: argparse.Namespace) -> argparse.Namespace:
    yaml_path = Path(args.source_args_yaml) if args.source_args_yaml else Path(args.checkpoint).parent.parent / "args.yaml"
    if not yaml_path.exists():
        return args

    with yaml_path.open("r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f) or {}

    keys = [
        "arch",
        "dataset",
        "img_size",
        "patch_size",
        "overlap",
        "scales",
        "mean",
        "std",
        "feat_dim",
        "mil_type",
        "pooling_type",
        "type_mil_encoder",
        "fcl_attention_dim",
        "map_prob_func",
        "fcl_encoder_dim",
        "sab_num_heads",
        "isab_num_heads",
        "pma_num_heads",
        "num_encoder_blocks",
        "trans_num_inds",
        "trans_layer_norm",
        "multi_scale_model",
        "fpn_dim",
        "upsample_method",
        "norm_fpn",
        "deep_supervision",
        "type_scale_aggregator",
        "nested_model",
        "type_region_aggregator",
        "type_region_encoder",
        "type_region_pooling",
        "drop_classhead",
        "drop_attention_pool",
        "drop_mha",
        "fcl_dropout",
    ]
    for key in keys:
        if key in train_args:
            setattr(args, key, train_args[key])
    return args


def parse_cohorts(spec: str) -> set[int]:
    cohorts: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            cohorts.update(range(int(left), int(right) + 1))
        else:
            cohorts.add(int(part))
    return cohorts


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args = maybe_load_training_args(args)
    args.checkpoint = Path(args.checkpoint)
    args.data_dir = Path(args.data_dir)
    args.out_dir = Path(args.out_dir)
    args.img_dir = Path(args.img_dir)
    args.clip_chk_pt_path = str(Path(args.clip_chk_pt_path))
    args.n_class = 1
    args.num_classes = 1
    args.model_type = "Classifier"
    args.train = False
    args.roi_eval = False
    args.feature_extraction = "online"
    args.data_aug = False
    args.apex = True
    args.training_mode = "frozen"
    args.warmup_stage_epochs = 0
    args.output_dir = str(args.out_dir)
    args.num_workers = 0
    if isinstance(args.img_size, tuple):
        args.img_size = list(args.img_size)
    if isinstance(args.scales, tuple):
        args.scales = list(args.scales)
    if isinstance(args.overlap, (float, int)):
        args.overlap = [float(args.overlap)]
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return args


def pad_image(img_array: torch.Tensor, patch_size: int, mean: float, std: float):
    _, h, w = img_array.shape
    new_h = h + (patch_size - h % patch_size)
    new_w = w + (patch_size - w % patch_size)
    additional_h = new_h - h
    additional_w = new_w - w

    horizontal_sum = img_array.sum(axis=(0, 1))
    left_info = horizontal_sum[: w // 2].sum()
    right_info = horizontal_sum[w // 2 :].sum()
    padding_left, padding_right = (additional_w, 0) if left_info < right_info else (0, additional_w)

    vertical_sum = img_array.sum(axis=(0, 2))
    top_info = vertical_sum[: h // 2].sum()
    bottom_info = vertical_sum[h // 2 :].sum()
    padding_top, padding_bottom = (additional_h, 0) if top_info < bottom_info else (0, additional_h)

    normalized_black_value = (0.0 - mean) / std
    padded = F.pad(
        img_array,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=normalized_black_value,
    )
    return padded, (padding_left, padding_right, padding_top, padding_bottom)


def image_to_tensor(img: Image.Image, mean: float, std: float) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - float(mean)) / float(std)


def extract_patches(img_tensor: torch.Tensor, args: argparse.Namespace):
    padded, padding = pad_image(img_tensor, args.patch_size, args.mean, args.std)
    _, img_h, img_w = padded.shape
    step_size = args.patch_size - int(args.patch_size * args.overlap[0])
    stop_y = min(img_h, img_h - args.patch_size + 1)
    stop_x = min(img_w, img_w - args.patch_size + 1)
    x_range = np.arange(0, stop_x, step=step_size)
    y_range = np.arange(0, stop_y, step=step_size)

    patches = []
    coords = []
    for x in x_range:
        for y in y_range:
            x_int = int(x)
            y_int = int(y)
            patches.append(padded[:, y_int : y_int + args.patch_size, x_int : x_int + args.patch_size])
            coords.append([x_int, y_int])

    coords_np = np.asarray(coords, dtype=np.int32)
    sort_idx = np.lexsort((coords_np[:, 0], coords_np[:, 1]))
    return torch.stack(patches)[sort_idx], coords_np[sort_idx], padding, (img_h, img_w)


def unnormalize_for_display(padded: torch.Tensor, mean: float, std: float) -> np.ndarray:
    img = (padded * float(std) + float(mean)).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def make_mask(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    threshold = max(0.02, float(np.percentile(gray, 5)))
    mask = gray > threshold
    if mask.sum() < 100:
        mask = gray > 0
    return mask


def normalize_map(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None and mask.any():
        valid = arr[mask]
    else:
        valid = arr.reshape(-1)
    vmin = float(valid.min()) if valid.size else float(arr.min())
    vmax = float(valid.max()) if valid.size else float(arr.max())
    if vmax - vmin < 1e-8:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = (arr - vmin) / (vmax - vmin)
    if mask is not None:
        out = np.where(mask, out, 0.0)
    return out.astype(np.float32)


def blur_map(heatmap: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return heatmap
    try:
        import cv2

        kernel = max(3, int(round(sigma * 6)) | 1)
        return cv2.GaussianBlur(heatmap, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    except Exception:
        return heatmap


def compute_heatmaps(model, patches, coords, padded_shape, mask, args, device):
    patches = patches.unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda" and args.dtype == "float16"):
            output = model(patches)
        if isinstance(output, (list, tuple)):
            output = output[0]
        score = float(torch.sigmoid(output.reshape(-1)[0]).detach().cpu())

    patch_scores = model.get_patch_scores()
    scale_scores = model.get_scale_scores()
    if scale_scores is not None:
        scale_weights = scale_scores.detach().float().cpu().reshape(-1).numpy()
        if len(scale_weights) != len(args.scales):
            scale_weights = np.ones(len(args.scales), dtype=np.float32) / max(1, len(args.scales))
    else:
        scale_weights = np.ones(len(args.scales), dtype=np.float32) / max(1, len(args.scales))

    img_h, img_w = padded_shape
    heatmaps = {}
    aggregated = np.zeros((img_h, img_w), dtype=np.float32)

    for idx, scale in enumerate(args.scales):
        if scale not in patch_scores:
            continue
        attention = patch_scores[scale].detach().float().cpu().squeeze()
        ratio = int(math.ceil(args.patch_size / scale))
        attention = attention.reshape(len(coords), ratio, ratio)

        attention_map = np.zeros((img_h, img_w), dtype=np.float32)
        count_map = np.zeros((img_h, img_w), dtype=np.float32)
        for patch_idx, (x, y) in enumerate(coords):
            x_start, y_start = int(x), int(y)
            x_end = min(img_w, x_start + args.patch_size)
            y_end = min(img_h, y_start + args.patch_size)
            patch_map = attention[patch_idx].unsqueeze(0).unsqueeze(0)
            patch_map = F.interpolate(
                patch_map,
                size=(args.patch_size, args.patch_size),
                mode="bilinear",
                align_corners=True,
            ).squeeze().numpy()
            patch_map = normalize_map(patch_map)
            attention_map[y_start:y_end, x_start:x_end] += patch_map[: y_end - y_start, : x_end - x_start]
            count_map[y_start:y_end, x_start:x_end] += 1.0

        heatmap = np.divide(attention_map, np.maximum(count_map, 1.0))
        heatmap = blur_map(heatmap, args.sigma)
        heatmap = normalize_map(heatmap, mask)
        heatmaps[str(scale)] = heatmap
        aggregated += heatmap * float(scale_weights[idx])

    heatmaps["aggregated"] = normalize_map(aggregated, mask)
    return score, scale_weights, heatmaps


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    csv_path = args.data_dir / args.csv_file
    df = pd.read_csv(csv_path).fillna(0)
    if args.label not in df.columns:
        raise ValueError(f"Label column {args.label!r} not found in {csv_path}.")
    if args.split != "all":
        cohorts = parse_cohorts(args.test_cohorts if args.split == "test" else args.train_cohorts)
        if "cohort_num" not in df.columns:
            raise ValueError("CSV must contain cohort_num when --split is train/test.")
        df = df[df["cohort_num"].astype(int).isin(cohorts)]
    if args.only_positive:
        df = df[pd.to_numeric(df[args.label], errors="coerce").fillna(0).astype(int) == 1]
    if args.patient_id is not None:
        df = df[df["patient_id"].astype(str) == str(args.patient_id)]
    if args.image_id is not None:
        df = df[df["image_id"].astype(str) == str(args.image_id)]
    return df.reset_index(drop=True)


def load_uncertainty_lookup(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, float]]:
    if args.uncertainty_csv is None:
        return {}
    path = Path(args.uncertainty_csv)
    pred_df = pd.read_csv(path).fillna(0)
    required = {"patient_id", "image_id", args.uncertainty_col}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    if "split" in pred_df.columns:
        pred_df = pred_df[pred_df["split"].astype(str).str.lower().eq(args.split)]

    agg_dict = {args.uncertainty_col: "mean"}
    if args.uncertainty_score_col in pred_df.columns:
        agg_dict[args.uncertainty_score_col] = "mean"
    pred_df = pred_df.groupby(["patient_id", "image_id"], as_index=False).agg(agg_dict)

    lookup = {}
    for _, row in pred_df.iterrows():
        key = (str(row["patient_id"]), str(row["image_id"]))
        lookup[key] = {
            "uncertainty": float(row[args.uncertainty_col]),
            "uncertainty_score": float(row[args.uncertainty_score_col])
            if args.uncertainty_score_col in row
            else float("nan"),
        }
    return lookup


def save_heatmap_grid(stem: Path, rgb: np.ndarray, heatmaps: dict[str, np.ndarray], score: float, row, args):
    keys = [str(scale) for scale in args.scales if str(scale) in heatmaps] + ["aggregated"]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4.0 * (len(keys) + 1), 4.2))
    axes[0].imshow(rgb, cmap="gray")
    axes[0].set_title(f"image\nscore={score:.3f}")
    axes[0].axis("off")
    for ax, key in zip(axes[1:], keys):
        im = ax.imshow(heatmaps[key], cmap=args.cmap, vmin=0, vmax=1)
        ax.set_title(f"{key}")
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.suptitle(f"patient={row['patient_id']} image={row['image_id']}", fontsize=10)
    fig.savefig(stem.with_suffix(".heatmap_grid.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overlay_grid(stem: Path, rgb: np.ndarray, heatmaps: dict[str, np.ndarray], score: float, row, args):
    keys = [str(scale) for scale in args.scales if str(scale) in heatmaps] + ["aggregated"]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4.0 * (len(keys) + 1), 4.2))
    axes[0].imshow(rgb, cmap="gray")
    axes[0].set_title(f"image\nscore={score:.3f}")
    axes[0].axis("off")
    for ax, key in zip(axes[1:], keys):
        ax.imshow(rgb, cmap="gray")
        ax.imshow(heatmaps[key], cmap=args.cmap, alpha=args.overlay_alpha, vmin=0, vmax=1)
        ax.set_title(f"{key}")
        ax.axis("off")
    fig.suptitle(f"patient={row['patient_id']} image={row['image_id']}", fontsize=10)
    fig.savefig(stem.with_suffix(".overlay_grid.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_uncertainty_grid(
    stem: Path,
    rgb: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    uncertainty: float,
    row,
    args,
):
    keys = [str(scale) for scale in args.scales if str(scale) in heatmaps] + ["aggregated"]
    maps = {key: normalize_map(heatmaps[key] * float(uncertainty)) for key in keys}
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4.0 * (len(keys) + 1), 4.2))
    axes[0].imshow(rgb, cmap="gray")
    axes[0].set_title(f"image\nunc={uncertainty:.3f}")
    axes[0].axis("off")
    for ax, key in zip(axes[1:], keys):
        im = ax.imshow(maps[key], cmap=args.cmap, vmin=0, vmax=1)
        ax.set_title(f"{key}")
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.suptitle(f"attention-guided uncertainty: patient={row['patient_id']} image={row['image_id']}", fontsize=10)
    fig.savefig(stem.with_suffix(".uncertainty_grid.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = normalize_args(parse_args())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from MIL import build_model

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_model(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_msg = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model.is_training = False
    for param in model.parameters():
        param.requires_grad = False

    candidates = load_candidates(args)
    uncertainty_lookup = load_uncertainty_lookup(args)
    rows = []
    pending = []

    for _, row in candidates.iterrows():
        image_path = args.data_dir / args.img_dir / str(row["patient_id"]) / str(row["image_id"])
        if not image_path.exists():
            continue
        pending.append((row, image_path))
        if not args.top_score and len(pending) >= args.max_images:
            break

    if args.top_score:
        scored = []
        for row, image_path in pending:
            img = Image.open(image_path).convert("RGB")
            tensor = image_to_tensor(img, args.mean, args.std)
            patches, coords, padding, padded_shape = extract_patches(tensor, args)
            padded, _ = pad_image(tensor, args.patch_size, args.mean, args.std)
            rgb = unnormalize_for_display(padded, args.mean, args.std)
            mask = make_mask(rgb)
            score, scale_weights, heatmaps = compute_heatmaps(model, patches, coords, padded_shape, mask, args, device)
            if args.min_score is None or score >= args.min_score:
                scored.append((score, row, image_path, rgb, heatmaps, scale_weights))
        scored.sort(key=lambda item: item[0], reverse=True)
        work_items = scored[: args.max_images]
    else:
        work_items = []
        for row, image_path in pending:
            img = Image.open(image_path).convert("RGB")
            tensor = image_to_tensor(img, args.mean, args.std)
            patches, coords, padding, padded_shape = extract_patches(tensor, args)
            padded, _ = pad_image(tensor, args.patch_size, args.mean, args.std)
            rgb = unnormalize_for_display(padded, args.mean, args.std)
            mask = make_mask(rgb)
            score, scale_weights, heatmaps = compute_heatmaps(model, patches, coords, padded_shape, mask, args, device)
            if args.min_score is None or score >= args.min_score:
                work_items.append((score, row, image_path, rgb, heatmaps, scale_weights))
            if len(work_items) >= args.max_images:
                break

    for index, (score, row, image_path, rgb, heatmaps, scale_weights) in enumerate(work_items):
        safe_image = Path(str(row["image_id"])).stem[:80]
        stem = args.out_dir / f"{index:03d}_p{row['patient_id']}_{safe_image}"
        save_heatmap_grid(stem, rgb, heatmaps, score, row, args)
        if not args.no_overlay:
            save_overlay_grid(stem, rgb, heatmaps, score, row, args)

        uncertainty_info = uncertainty_lookup.get((str(row["patient_id"]), str(row["image_id"])))
        if uncertainty_info is not None:
            save_uncertainty_grid(stem, rgb, heatmaps, uncertainty_info["uncertainty"], row, args)

        if args.save_npy:
            np.savez_compressed(stem.with_suffix(".heatmaps.npz"), **heatmaps)

        rows.append(
            {
                "index": index,
                "patient_id": row["patient_id"],
                "image_id": row["image_id"],
                "label": int(row[args.label]),
                "cohort_num": row.get("cohort_num", ""),
                "prediction_score": score,
                "dst_edl_uncertainty": ""
                if uncertainty_info is None
                else uncertainty_info["uncertainty"],
                "dst_edl_prediction_score": ""
                if uncertainty_info is None
                else uncertainty_info["uncertainty_score"],
                "scale_weights": json.dumps([float(x) for x in scale_weights]),
                "image_path": str(image_path),
                "heatmap_grid": str(stem.with_suffix(".heatmap_grid.png")),
                "overlay_grid": "" if args.no_overlay else str(stem.with_suffix(".overlay_grid.png")),
                "uncertainty_grid": ""
                if uncertainty_info is None
                else str(stem.with_suffix(".uncertainty_grid.png")),
            }
        )

    pd.DataFrame(rows).to_csv(args.out_dir / "manifest.csv", index=False)
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "load_state_dict": str(load_msg),
                "note": "MIL attention heatmaps only; not EDL/DST uncertainty maps.",
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            f,
            indent=2,
        )
    print(f"Saved {len(rows)} heatmap examples to {args.out_dir}")


if __name__ == "__main__":
    main()
