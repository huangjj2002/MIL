"""Export reusable origin/Mammo-CLIP encoder embeddings.

The default output is patch-level image-encoder embeddings stored as one
continuous embeddings.npy array, plus metadata.csv and manifest.json. These
features are intended as a fast input cache for later EDL, DST, or
prototype-style modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RGB_ARCHES = {
    "upmc_breast_clip_det_b5_period_n_ft",
    "upmc_vindr_breast_clip_det_b5_period_n_ft",
    "upmc_breast_clip_det_b5_period_n_lp",
    "upmc_vindr_breast_clip_det_b5_period_n_lp",
    "upmc_breast_clip_det_b2_period_n_ft",
    "upmc_vindr_breast_clip_det_b2_period_n_ft",
    "upmc_breast_clip_det_b2_period_n_lp",
    "upmc_vindr_breast_clip_det_b2_period_n_lp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export origin/Mammo-CLIP encoder embeddings to embeddings.npy "
            "with metadata.csv and manifest.json for fast downstream EDL/DST training."
        )
    )

    # Data and output paths
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="datasets/Vindir-mammoclip", type=str)
    parser.add_argument(
        "--img-dir",
        "--img_dir",
        dest="img_dir",
        default="VinDir_preprocessed_mammoclip/images_png",
        type=str,
        help="Image directory relative to data-dir.",
    )
    parser.add_argument("--csv-file", "--csv_file", dest="csv_file", default="grouped_df.csv", type=str)
    parser.add_argument(
        "--csv-path",
        "--csv_path",
        dest="csv_path",
        default=None,
        type=str,
        help="Optional direct CSV path. Overrides data-dir/csv-file when provided.",
    )
    parser.add_argument(
        "--img-root",
        "--img_root",
        dest="img_root",
        default=None,
        type=str,
        help="Optional direct image root. Overrides data-dir/img-dir when provided.",
    )
    parser.add_argument("--clip_chk_pt_path", required=True, type=str)
    parser.add_argument("--out-dir", default="origin_encoder_embeddings", type=str)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )

    # Split and metadata
    parser.add_argument("--label", default="Mass", type=str)
    parser.add_argument("--train-cohorts", "--train_cohorts", dest="train_cohorts", default="1-8", type=str)
    parser.add_argument("--test-cohorts", "--test_cohorts", dest="test_cohorts", default="9-10", type=str)
    parser.add_argument("--val-split", "--val_split", dest="val_split", default=0.2, type=float)
    parser.add_argument("--val-max-frac", "--val_max_frac", dest="val_max_frac", default=0.5, type=float)
    parser.add_argument("--data-frac", "--data_frac", dest="data_frac", default=1.0, type=float)
    parser.add_argument(
        "--max-samples",
        default=None,
        type=int,
        help="Optional maximum number of rows exported per split, useful for smoke tests.",
    )
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument(
        "--prototype-k",
        "--prototype_k",
        dest="prototype_k",
        default=10,
        type=int,
        help=(
            "For bag_origin exports, select K real train samples per class as "
            "downstream DST prototypes. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--prototype-split",
        "--prototype_split",
        dest="prototype_split",
        default="train",
        choices=["train", "train_val", "all"],
        help="Rows eligible for prototype selection during bag_origin export.",
    )
    parser.add_argument(
        "--prototype-normal-views-only",
        "--prototype_normal_views_only",
        dest="prototype_normal_views_only",
        action="store_true",
        default=False,
        help=(
            "When selecting bag_origin prototypes, keep only standard mammography "
            "views when a view-like metadata column is available."
        ),
    )
    parser.add_argument(
        "--prototype-view-col",
        "--prototype_view_col",
        dest="prototype_view_col",
        default=None,
        type=str,
        help="Optional metadata column used to identify CC/MLO views for prototype selection.",
    )
    parser.add_argument(
        "--prototype-allowed-views",
        "--prototype_allowed_views",
        dest="prototype_allowed_views",
        nargs="*",
        default=["CC", "MLO", "LCC", "RCC", "LMLO", "RMLO"],
        help="Allowed view labels for --prototype-normal-views-only.",
    )
    parser.add_argument(
        "--prototype-exclude-view-keywords",
        "--prototype_exclude_view_keywords",
        dest="prototype_exclude_view_keywords",
        nargs="*",
        default=[
            "spot",
            "magnification",
            "mag",
            "compression",
            "implant",
            "displaced",
            "rolled",
            "tangent",
            "tangential",
            "cleavage",
            "axillary",
        ],
        help=(
            "Case-insensitive metadata keywords excluded from prototype selection "
            "when --prototype-normal-views-only is enabled."
        ),
    )

    # Image and encoder settings
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str)
    parser.add_argument("--dataset", default="ViNDr", type=str)
    parser.add_argument("--img-size", "--img_size", dest="img_size", nargs="+", type=int, default=[1520, 912])
    parser.add_argument("--patch_size", "--patch-size", dest="patch_size", default=512, type=int)
    parser.add_argument("--overlap", nargs="*", default=[0.0], type=float)
    parser.add_argument("--scales", nargs="*", default=[16, 32, 128], type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", default=None, type=str)
    parser.add_argument(
        "--encoder-batch-size",
        "--encoder_batch_size",
        dest="encoder_batch_size",
        default=64,
        type=int,
        help="Number of image patches per forward pass through the image encoder.",
    )
    parser.add_argument("--amp", default="y", choices=["y", "n"], help="Use CUDA autocast while exporting.")

    # Save format
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument(
        "--preview-samples",
        "--preview_samples",
        dest="preview_samples",
        default=0,
        type=int,
        help=(
            "When exporting patch_encoder embeddings, also save a small "
            "patch_embedding_preview.npz/csv for the first N samples."
        ),
    )
    parser.add_argument(
        "--preview-max-patches",
        "--preview_max_patches",
        dest="preview_max_patches",
        default=0,
        type=int,
        help=(
            "Maximum patches per preview sample. Use 0 to keep all patches "
            "for previewed samples."
        ),
    )

    # Embedding level
    parser.add_argument(
        "--embedding-level",
        "--embedding_level",
        dest="embedding_level",
        default="patch_encoder",
        choices=["patch_encoder", "bag_origin", "origin_patch"],
        help=(
            "patch_encoder exports Mammo-CLIP image-encoder patch embeddings; "
            "bag_origin exports origin MIL classifier-input bag embeddings; "
            "origin_patch exports origin MIL encoded patch/token embeddings before MIL pooling."
        ),
    )
    parser.add_argument(
        "--origin-checkpoint",
        "--origin_checkpoint",
        dest="origin_checkpoint",
        default=None,
        type=str,
        help="Origin MIL best_model.pth; required for --embedding-level bag_origin or origin_patch.",
    )

    # Minimal origin MIL model args for optional bag_origin export.
    parser.add_argument("--mil_type", "--mil-type", dest="mil_type", default="pyramidal_mil", choices=["embedding", "pyramidal_mil"])
    parser.add_argument("--pooling_type", "--pooling-type", dest="pooling_type", default="mean", choices=["max", "mean", "attention", "gated-attention", "pma"])
    parser.add_argument("--type_mil_encoder", "--type-mil-encoder", dest="type_mil_encoder", default="mlp", choices=["mlp", "sab", "isab"])
    parser.add_argument("--fcl_attention_dim", "--fcl-attention-dim", dest="fcl_attention_dim", default=128, type=int)
    parser.add_argument("--map_prob_func", "--map-prob-func", dest="map_prob_func", default="softmax", choices=["softmax", "sparsemax", "entmax", "alpha_entmax"])
    parser.add_argument("--fcl_encoder_dim", "--fcl-encoder-dim", dest="fcl_encoder_dim", default=256, type=int)
    parser.add_argument("--sab_num_heads", "--sab-num-heads", dest="sab_num_heads", default=4, type=int)
    parser.add_argument("--isab_num_heads", "--isab-num-heads", dest="isab_num_heads", default=4, type=int)
    parser.add_argument("--pma_num_heads", "--pma-num-heads", dest="pma_num_heads", default=1, type=int)
    parser.add_argument("--num_encoder_blocks", "--num-encoder-blocks", dest="num_encoder_blocks", default=2, type=int)
    parser.add_argument("--trans_num_inds", "--trans-num-inds", dest="trans_num_inds", default=20, type=int)
    parser.add_argument("--trans_layer_norm", "--trans-layer-norm", dest="trans_layer_norm", action="store_true", default=False)
    parser.add_argument("--multi_scale_model", "--multi-scale-model", dest="multi_scale_model", default="fpn", choices=["none", "fpn", "backbone_pyramid", "msp"])
    parser.add_argument("--fpn_dim", "--fpn-dim", dest="fpn_dim", default=256, type=int)
    parser.add_argument("--upsample_method", "--upsample-method", dest="upsample_method", default="nearest", choices=["bilinear", "nearest"])
    parser.add_argument("--norm_fpn", "--norm-fpn", dest="norm_fpn", action="store_true", default=False)
    parser.add_argument("--deep_supervision", "--deep-supervision", dest="deep_supervision", action="store_true", default=False)
    parser.add_argument(
        "--type_scale_aggregator",
        "--type-scale-aggregator",
        dest="type_scale_aggregator",
        default="concatenation",
        choices=["concatenation", "max_p", "mean_p", "attention", "gated-attention"],
    )
    parser.add_argument("--nested_model", "--nested-model", dest="nested_model", action="store_true", default=False)
    parser.add_argument("--type_region_aggregator", "--type-region-aggregator", dest="type_region_aggregator", default=None)
    parser.add_argument("--type_region_encoder", "--type-region-encoder", dest="type_region_encoder", default="none", choices=["none", "mlp", "sab", "isab"])
    parser.add_argument("--type_region_pooling", "--type-region-pooling", dest="type_region_pooling", default="none", choices=["none", "max", "mean", "attention", "gated-attention", "pma"])
    parser.add_argument("--feat_dim", "--feat-dim", dest="feat_dim", default=352, type=int)
    parser.add_argument("--drop_classhead", "--drop-classhead", dest="drop_classhead", default=0.0, type=float)
    parser.add_argument("--drop_attention_pool", "--drop-attention-pool", dest="drop_attention_pool", default=0.0, type=float)
    parser.add_argument("--drop_mha", "--drop-mha", dest="drop_mha", default=0.0, type=float)
    parser.add_argument("--fcl_dropout", "--fcl-dropout", dest="fcl_dropout", default=0.0, type=float)

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_dir = Path(args.data_dir)
    args.out_dir = Path(args.out_dir)
    args.img_dir = Path(args.img_dir)
    args.csv_file = Path(args.csv_file)
    args.csv_path = Path(args.csv_path) if args.csv_path is not None else None
    args.img_root = Path(args.img_root) if args.img_root is not None else None
    args.clip_chk_pt_path = str(Path(args.clip_chk_pt_path))
    if args.origin_checkpoint is not None:
        args.origin_checkpoint = str(Path(args.origin_checkpoint))

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.multi_scale_model == "none":
        args.multi_scale_model = None
    if args.type_region_encoder == "none":
        args.type_region_encoder = None
    if args.type_region_pooling == "none":
        args.type_region_pooling = None

    if len(args.img_size) != 2:
        raise ValueError(f"--img-size expects two integers [H W], got: {args.img_size}")
    if not args.overlap:
        args.overlap = [0.0]
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.encoder_batch_size <= 0:
        raise ValueError("--encoder-batch-size must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    if args.preview_samples < 0:
        raise ValueError("--preview-samples must be non-negative.")
    if args.preview_max_patches < 0:
        raise ValueError("--preview-max-patches must be non-negative.")
    if args.prototype_k < 0:
        raise ValueError("--prototype-k must be non-negative.")
    if args.embedding_level in {"bag_origin", "origin_patch"} and args.origin_checkpoint is None:
        raise ValueError("--origin-checkpoint is required when --embedding-level bag_origin or origin_patch.")
    if args.embedding_level in {"bag_origin", "origin_patch"} and args.multi_scale_model == "msp":
        raise ValueError(f"{args.embedding_level} export does not support multi_scale_model=msp in this script.")

    # Attributes consumed by the existing MIL builders.
    args.feature_extraction = "online"
    args.train = False
    args.n_class = 1
    args.num_classes = 1
    args.model_type = "Classifier"
    args.data_aug = False
    args.roi_eval = False
    args.apex = args.amp == "y"
    args.training_mode = "frozen"
    args.warmup_stage_epochs = 0

    return args


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def collate_single(batch):
    return batch[0]


def sanitize_key(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def make_sample_key(row, used_keys: set[str]) -> str:
    source_index = int(row["source_index"]) if "source_index" in row else len(used_keys)
    base_key = (
        f"idx{source_index:06d}_"
        f"p{sanitize_key(row.get('patient_id', 'unknown'))}_"
        f"i{sanitize_key(row.get('image_id', 'unknown'))}"
    )
    sample_key = base_key
    suffix = 1
    while sample_key in used_keys:
        sample_key = f"{base_key}_{suffix}"
        suffix += 1
    used_keys.add(sample_key)
    return sample_key


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    return value


def as_attr_value(value):
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(to_jsonable(value))
    return value


def get_device(args):
    import torch

    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_image_like_training(img_tensor, patch_size, mean, std):
    import torch.nn.functional as F

    if len(img_tensor.size()) == 3:
        _, height, width = img_tensor.size()
    else:
        height, width = img_tensor.size()

    new_height = height + (patch_size - height % patch_size)
    new_width = width + (patch_size - width % patch_size)
    additional_height = new_height - height
    additional_width = new_width - width

    padding_left = padding_right = padding_top = padding_bottom = 0

    horizontal_sum = img_tensor.sum(axis=(0, 1))
    left_info = horizontal_sum[: width // 2].sum()
    right_info = horizontal_sum[width // 2 :].sum()
    if left_info < right_info:
        padding_left = additional_width
    else:
        padding_right = additional_width

    vertical_sum = img_tensor.sum(axis=(0, 2))
    top_info = vertical_sum[: height // 2].sum()
    bottom_info = vertical_sum[height // 2 :].sum()
    if top_info < bottom_info:
        padding_top = additional_height
    else:
        padding_bottom = additional_height

    normalized_black_value = (0.0 - mean) / std
    padded_img = F.pad(
        img_tensor,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=normalized_black_value,
    )
    return padded_img, (padding_left, padding_right, padding_top, padding_bottom)


def extract_fixed_patches(img_tensor, patch_size, overlap):
    import numpy as np
    import torch

    _, img_height, img_width = img_tensor.shape
    step_size = patch_size - int(patch_size * float(overlap[0]))
    if step_size <= 0:
        raise ValueError(f"Invalid overlap {overlap[0]} creates non-positive step size.")

    stop_y = min(img_height, img_height - patch_size + 1)
    stop_x = min(img_width, img_width - patch_size + 1)
    x_range = np.arange(0, stop_x, step=step_size)
    y_range = np.arange(0, stop_y, step=step_size)

    patches = []
    coords = []
    for x_start in x_range:
        for y_start in y_range:
            x_int = int(x_start)
            y_int = int(y_start)
            patches.append(img_tensor[:, y_int : y_int + patch_size, x_int : x_int + patch_size])
            coords.append([x_int, y_int])

    if not patches:
        raise RuntimeError(
            f"No patches extracted from padded image with shape {(img_height, img_width)} "
            f"and patch_size={patch_size}."
        )

    coords = np.asarray(coords, dtype=np.int32)
    patches = torch.stack(patches)
    sorted_indices = np.lexsort((coords[:, 0], coords[:, 1]))
    return patches[sorted_indices], coords[sorted_indices]


class FixedPatchBagDataset:
    def __init__(self, args: argparse.Namespace, df):
        import torchvision.transforms as transforms

        self.args = args
        self.df = df.reset_index(drop=True)
        self.img_root = args.img_root if args.img_root is not None else args.data_dir / args.img_dir
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=args.mean, std=args.std)
        self.use_rgb = args.arch.lower() in RGB_ARCHES

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, row) -> Path:
        if "image_file_path" in row and row["image_file_path"]:
            direct_path = Path(str(row["image_file_path"]))
            if direct_path.exists():
                return direct_path

        image_path = self.img_root / str(row["patient_id"]) / str(row["image_id"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(row)
        image = Image.open(image_path)
        image = image.convert("RGB" if self.use_rgb else "L")

        tensor = self.normalize(self.to_tensor(image))
        padded_tensor, padding = pad_image_like_training(
            tensor,
            patch_size=self.args.patch_size,
            mean=self.args.mean,
            std=self.args.std,
        )
        patches, coords = extract_fixed_patches(
            padded_tensor,
            patch_size=self.args.patch_size,
            overlap=self.args.overlap,
        )

        return {
            "x": patches,
            "coords": coords,
            "padding": padding,
            "row": row.to_dict(),
            "image_path": str(image_path),
        }


def load_and_split_dataframe(args: argparse.Namespace):
    import pandas as pd

    from utils.data_split_utils import adaptive_stratified_train_val_split, split_df_by_cohorts

    csv_path = args.csv_path if args.csv_path is not None else args.data_dir / args.csv_file
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path).fillna(0)
    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' was not found in {csv_path}.")
    for required_col in ["patient_id", "image_id"]:
        if required_col not in df.columns:
            raise ValueError(f"Required column '{required_col}' was not found in {csv_path}.")

    df = df.reset_index(drop=False).rename(columns={"index": "source_index"})
    _, dev_df, test_df = split_df_by_cohorts(
        df,
        train_cohorts=args.train_cohorts,
        test_cohorts=args.test_cohorts,
    )

    if args.data_frac < 1.0:
        dev_df = dev_df.sample(frac=args.data_frac, random_state=args.seed, ignore_index=True)

    split_args = SimpleNamespace(label=args.label, seed=args.seed)
    train_df, val_df = adaptive_stratified_train_val_split(
        dev_df,
        val_frac=args.val_split,
        max_val_frac=args.val_max_frac,
        args=split_args,
        context="Embedding export train/val split",
    )

    split_dfs = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    if args.max_samples is not None:
        split_dfs = {
            split: split_df.head(args.max_samples).reset_index(drop=True)
            for split, split_df in split_dfs.items()
        }

    return split_dfs


def load_image_encoder(args: argparse.Namespace, device):
    import torch

    from FeatureExtractors.mammoclip import load_image_encoder as build_image_encoder

    checkpoint = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    encoder_config = checkpoint["config"]["model"]["image_encoder"]
    image_encoder = build_image_encoder(encoder_config, multi_scale=False)

    image_encoder_weights = {}
    for key, value in checkpoint["model"].items():
        if key.startswith("image_encoder."):
            image_encoder_weights[".".join(key.split(".")[1:])] = value
    load_msg = image_encoder.load_state_dict(image_encoder_weights, strict=False)

    for param in image_encoder.parameters():
        param.requires_grad = False
    image_encoder.to(device)
    image_encoder.eval()
    return image_encoder, encoder_config, str(load_msg)


def run_patch_encoder(image_encoder, patches, args: argparse.Namespace, device):
    import numpy as np
    import torch

    features = []
    amp_enabled = args.apex and device.type == "cuda"
    with torch.no_grad():
        for start_idx in range(0, patches.size(0), args.encoder_batch_size):
            patch_batch = patches[start_idx : start_idx + args.encoder_batch_size].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                encoded = image_encoder(patch_batch)
            if isinstance(encoded, (list, tuple)):
                if len(encoded) != 1:
                    raise RuntimeError(
                        "Patch encoder returned multiple outputs. Use --embedding-level bag_origin "
                        "for origin FPN embeddings, or keep patch_encoder with the default vector image encoder."
                    )
                encoded = encoded[0]
            if encoded.ndim > 2:
                encoded = encoded.flatten(start_dim=1)
            features.append(encoded.detach().float().cpu())

    feature_tensor = torch.cat(features, dim=0)
    target_dtype = np.float16 if args.dtype == "float16" else np.float32
    return feature_tensor.numpy().astype(target_dtype, copy=False)


def load_origin_mil_model(args: argparse.Namespace, device):
    import torch

    from MIL import build_model

    model = build_model(args)
    if not hasattr(model, "classifier"):
        raise ValueError(
            "bag_origin export requires an origin model with a final classifier. "
            "Use type_scale_aggregator=concatenation or gated-attention for pyramidal MIL."
        )

    checkpoint = torch.load(args.origin_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_msg = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model.is_training = False
    return model, str(load_msg)


def _binary_prob_from_origin_output(output):
    import torch

    if isinstance(output, (list, tuple)):
        output = output[0]
    output = output.detach().float()
    if output.ndim == 0:
        output = output.view(1, 1)
    elif output.ndim == 1:
        output = output.view(-1, 1)

    if output.size(-1) == 1:
        pos_prob = torch.sigmoid(output[:, 0])
        prob = torch.stack([1.0 - pos_prob, pos_prob], dim=-1)
    elif output.size(-1) == 2:
        prob = torch.softmax(output, dim=-1)
    else:
        raise RuntimeError(f"Expected binary origin model output, got shape {tuple(output.shape)}.")
    return prob


def run_bag_origin(model, patches, args: argparse.Namespace, device):
    import numpy as np
    import torch

    captured = []

    def capture_classifier_input(module, inputs):
        captured.append(inputs[0].detach().float().cpu())

    handle = model.classifier.register_forward_pre_hook(capture_classifier_input)
    amp_enabled = args.apex and device.type == "cuda"
    try:
        with torch.no_grad():
            inputs = patches.unsqueeze(0).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(inputs)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Could not capture bag_origin embedding from model.classifier input.")
    bag_embedding = captured[-1].reshape(1, -1)
    target_dtype = np.float16 if args.dtype == "float16" else np.float32
    prob = _binary_prob_from_origin_output(output).cpu().numpy()
    return bag_embedding.numpy().astype(target_dtype, copy=False), prob


def _origin_patch_hooks(model, args):
    captures = {}
    handles = []

    def make_hook(name):
        def capture_inputs(module, inputs):
            captures[name] = inputs[0].detach().float().cpu()
        return capture_inputs

    if hasattr(model, "side_inst_aggregator") and "aggregators" in model.side_inst_aggregator:
        for scale in args.scales:
            key = f"aggregator_{scale}"
            aggregators = model.side_inst_aggregator["aggregators"]
            if key in aggregators:
                handles.append(aggregators[key].register_forward_pre_hook(make_hook(scale)))
    elif hasattr(model, "aggregator"):
        handles.append(model.aggregator.register_forward_pre_hook(make_hook(args.patch_size)))
    else:
        raise ValueError(
            "Could not find a MIL aggregator to hook for origin_patch export. "
            "Expected model.aggregator or model.side_inst_aggregator['aggregators']."
        )

    if not handles:
        raise ValueError("No origin_patch hooks were registered. Check scales and MIL model configuration.")
    return captures, handles


def _origin_patch_token_rows(coords, scale, num_tokens, args):
    coords = np.asarray(coords, dtype=np.int32)
    if len(coords) == 0:
        return []

    if int(num_tokens) == len(coords):
        ratio = 1
    else:
        ratio = int(math.ceil(float(args.patch_size) / float(scale)))
        expected_tokens = len(coords) * ratio * ratio
        if expected_tokens != int(num_tokens):
            raise RuntimeError(
                f"Cannot map origin_patch tokens to coordinates for scale={scale}: "
                f"{num_tokens} tokens vs expected {expected_tokens} from "
                f"{len(coords)} patches and ratio={ratio}."
            )

    rows = []
    token_width = max(1, int(round(float(args.patch_size) / float(ratio))))
    for token_idx in range(int(num_tokens)):
        patch_idx = token_idx // (ratio * ratio)
        cell_idx = token_idx % (ratio * ratio)
        cell_y = cell_idx // ratio
        cell_x = cell_idx % ratio
        base_x, base_y = coords[patch_idx]
        rows.append(
            {
                "scale": int(scale),
                "patch_idx": int(patch_idx),
                "token_idx": int(token_idx),
                "cell_x": int(cell_x),
                "cell_y": int(cell_y),
                "x": int(base_x + cell_x * token_width),
                "y": int(base_y + cell_y * token_width),
                "token_width": int(token_width),
                "token_height": int(token_width),
            }
        )
    return rows


def run_origin_patch(model, patches, coords, args: argparse.Namespace, device):
    import numpy as np
    import torch

    captures, handles = _origin_patch_hooks(model, args)
    amp_enabled = args.apex and device.type == "cuda"
    try:
        with torch.no_grad():
            inputs = patches.unsqueeze(0).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    if not captures:
        raise RuntimeError("Could not capture origin_patch embeddings from MIL aggregators.")

    feature_chunks = []
    token_rows = []
    target_dtype = np.float16 if args.dtype == "float16" else np.float32
    scales_to_export = [scale for scale in args.scales if scale in captures]
    if not scales_to_export:
        scales_to_export = list(captures.keys())

    for scale in scales_to_export:
        if scale not in captures:
            continue
        tensor = captures[scale]
        if tensor.ndim != 3 or tensor.size(0) != 1:
            raise RuntimeError(
                f"Expected captured origin_patch tensor for scale={scale} to have "
                f"shape [1, N, D], got {tuple(tensor.shape)}."
            )
        scale_features = tensor.squeeze(0).numpy().astype(target_dtype, copy=False)
        feature_chunks.append(scale_features)
        token_rows.extend(
            _origin_patch_token_rows(
                coords=coords,
                scale=scale,
                num_tokens=scale_features.shape[0],
                args=args,
            )
        )

    if not feature_chunks:
        raise RuntimeError("No origin_patch feature chunks were captured.")

    features = np.concatenate(feature_chunks, axis=0)
    if len(token_rows) != int(features.shape[0]):
        raise RuntimeError(
            f"origin_patch metadata row count mismatch: {len(token_rows)} rows vs "
            f"{features.shape[0]} embeddings."
        )
    return features, token_rows


def build_metadata_row(
    row,
    split_name,
    sample_key,
    features,
    coords,
    args,
    image_path,
    padding,
    embedding_start,
    original_num_patches,
    origin_prob=None,
):
    metadata_row = dict(row)
    embedding_count = int(features.shape[0])
    metadata_row["export_split"] = split_name
    metadata_row["label"] = int(row[args.label])
    metadata_row["label_column"] = args.label
    metadata_row["cohort_num"] = int(row["cohort_num"])
    metadata_row["sample_key"] = sample_key
    metadata_row["embedding_file"] = "embeddings.npy"
    metadata_row["embedding_start"] = int(embedding_start)
    metadata_row["embedding_end"] = int(embedding_start + embedding_count)
    metadata_row["num_embeddings"] = embedding_count
    metadata_row["num_patches"] = int(original_num_patches)
    metadata_row["embedding_dim"] = int(features.shape[-1])
    metadata_row["dtype"] = str(features.dtype)
    metadata_row["embedding_level"] = args.embedding_level
    metadata_row["image_path"] = image_path
    metadata_row["coords"] = json.dumps(coords.astype(int).tolist(), separators=(",", ":"))
    metadata_row["padding_left"] = int(padding[0])
    metadata_row["padding_right"] = int(padding[1])
    metadata_row["padding_top"] = int(padding[2])
    metadata_row["padding_bottom"] = int(padding[3])
    if origin_prob is not None:
        prob = np.asarray(origin_prob, dtype=np.float32).reshape(-1)
        if prob.shape[0] >= 2:
            label_value = int(row[args.label])
            metadata_row["origin_prediction_score"] = float(prob[1])
            metadata_row["origin_predicted_class"] = int(np.argmax(prob))
            metadata_row["origin_true_class_score"] = float(prob[label_value])
    return metadata_row


def maybe_collect_patch_preview(preview_items, args, row, split_name, sample_key, features, coords, image_path):
    if args.embedding_level not in {"patch_encoder", "origin_patch"}:
        return
    if args.preview_samples <= 0 or len(preview_items) >= args.preview_samples:
        return

    keep_count = int(features.shape[0])
    if args.preview_max_patches > 0:
        keep_count = min(keep_count, args.preview_max_patches)

    preview_items.append(
        {
            "split": split_name,
            "sample_key": sample_key,
            "patient_id": str(row.get("patient_id", "")),
            "image_id": str(row.get("image_id", "")),
            "image_path": str(image_path),
            "patch_embeddings": features[:keep_count].copy(),
            "patch_coords": coords[:keep_count].astype(np.int32, copy=True),
            "total_patches": int(features.shape[0]),
            "preview_patches": int(keep_count),
        }
    )


def save_patch_preview(out_dir, preview_items):
    if not preview_items:
        return None

    patch_embeddings = []
    patch_coords = []
    patch_sample_index = []
    rows = []
    patch_offset = 0

    for sample_index, item in enumerate(preview_items):
        embeddings = item["patch_embeddings"]
        coords = item["patch_coords"]
        patch_count = int(embeddings.shape[0])
        patch_embeddings.append(embeddings)
        patch_coords.append(coords)
        patch_sample_index.append(
            np.full(patch_count, sample_index, dtype=np.int32)
        )
        rows.append(
            {
                "preview_sample_index": sample_index,
                "split": item["split"],
                "patient_id": item["patient_id"],
                "image_id": item["image_id"],
                "sample_key": item["sample_key"],
                "image_path": item["image_path"],
                "total_patches": item["total_patches"],
                "preview_patches": item["preview_patches"],
                "preview_patch_start": patch_offset,
                "preview_patch_end": patch_offset + patch_count,
            }
        )
        patch_offset += patch_count

    preview_npz_path = out_dir / "patch_embedding_preview.npz"
    preview_csv_path = out_dir / "patch_embedding_preview.csv"
    np.savez_compressed(
        preview_npz_path,
        patch_embeddings=np.concatenate(patch_embeddings, axis=0),
        patch_coords=np.concatenate(patch_coords, axis=0),
        patch_sample_index=np.concatenate(patch_sample_index, axis=0),
        split=np.asarray([item["split"] for item in preview_items], dtype=str),
        patient_id=np.asarray([item["patient_id"] for item in preview_items], dtype=str),
        image_id=np.asarray([item["image_id"] for item in preview_items], dtype=str),
        sample_key=np.asarray([item["sample_key"] for item in preview_items], dtype=str),
        image_path=np.asarray([item["image_path"] for item in preview_items], dtype=str),
    )

    import pandas as pd

    pd.DataFrame(rows).to_csv(preview_csv_path, index=False)
    return preview_npz_path, preview_csv_path


def _prototype_class_name(class_idx):
    return "negative" if int(class_idx) == 0 else "positive"


def _prototype_sort_key(df):
    correct = (
        df["origin_predicted_class"].astype(int).to_numpy()
        == df["label"].astype(int).to_numpy()
        if "origin_predicted_class" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    source_index = (
        df["source_index"].astype(int).to_numpy()
        if "source_index" in df.columns
        else np.arange(len(df), dtype=int)
    )
    return np.lexsort(
        (
            source_index,
            -df["selection_score"].astype(float).to_numpy(),
            -correct.astype(int),
        )
    )


def _normalize_view_text(value):
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _find_prototype_view_column(df, args):
    if args.prototype_view_col:
        if args.prototype_view_col not in df.columns:
            raise ValueError(
                f"--prototype-view-col {args.prototype_view_col!r} was not found in metadata columns."
            )
        return args.prototype_view_col

    preferred = [
        "view_position",
        "ViewPosition",
        "view",
        "View",
        "view_name",
        "image_view",
        "laterality_view",
        "projection",
        "SeriesDescription",
        "series_description",
        "ProtocolName",
        "protocol_name",
    ]
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for col in preferred:
        if col in df.columns:
            return col
        if col.lower() in lower_to_original:
            return lower_to_original[col.lower()]

    for col in df.columns:
        name = str(col).lower()
        if "view" in name or "projection" in name or "series" in name or "protocol" in name:
            return col
    return None


def _filter_prototype_normal_views(candidate_df, args):
    if not args.prototype_normal_views_only:
        return candidate_df

    view_col = _find_prototype_view_column(candidate_df, args)
    if view_col is None:
        print(
            "[WARN] --prototype-normal-views-only was set, but no view-like metadata "
            "column was found. Prototype selection will use all candidate rows."
        )
        return candidate_df

    allowed_views = {
        _normalize_view_text(view)
        for view in args.prototype_allowed_views
        if str(view).strip()
    }
    exclude_keywords = [
        str(keyword).strip().lower()
        for keyword in args.prototype_exclude_view_keywords
        if str(keyword).strip()
    ]

    text_cols = [
        col for col in candidate_df.columns
        if candidate_df[col].dtype == object or "view" in str(col).lower()
        or "series" in str(col).lower() or "protocol" in str(col).lower()
    ]
    if view_col not in text_cols:
        text_cols.append(view_col)

    keep_mask = []
    for _, row in candidate_df.iterrows():
        row_text = " ".join(str(row.get(col, "")) for col in text_cols).lower()
        has_excluded_keyword = any(keyword in row_text for keyword in exclude_keywords)
        normalized_view = _normalize_view_text(row.get(view_col, ""))
        has_allowed_view = (
            normalized_view in allowed_views
            or any(normalized_view.endswith(view) for view in allowed_views)
            or any(view in normalized_view for view in allowed_views if len(view) >= 3)
        )
        keep_mask.append(has_allowed_view and not has_excluded_keyword)

    filtered = candidate_df[np.asarray(keep_mask, dtype=bool)].copy()
    print(
        f"[INFO] Prototype normal-view filter using column {view_col!r}: "
        f"{len(filtered)}/{len(candidate_df)} candidates kept."
    )
    if filtered.empty:
        raise RuntimeError(
            "Prototype normal-view filter removed all candidate rows. "
            "Check --prototype-view-col / --prototype-allowed-views / "
            "--prototype-exclude-view-keywords."
        )
    return filtered


def _copy_prototype_image(row, args, class_idx, rank_idx):
    source_path = Path(str(row["image_path"]))
    if not source_path.exists():
        raise FileNotFoundError(f"Prototype source image not found: {source_path}")

    class_dir = args.out_dir / "prototype_images" / _prototype_class_name(class_idx)
    class_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix or ".png"
    filename = (
        f"c{class_idx}_p{rank_idx:02d}_"
        f"patient-{sanitize_key(row.get('patient_id', 'unknown'))}_"
        f"image-{sanitize_key(row.get('image_id', 'unknown'))}{suffix}"
    )
    target_path = class_dir / filename
    shutil.copy2(source_path, target_path)
    return target_path


def save_embedding_prototype_bank(args, metadata_df, embeddings):
    import pandas as pd

    if args.embedding_level != "bag_origin" or args.prototype_k <= 0:
        return None, None, None
    if metadata_df.empty:
        return None, None, None
    required_cols = {
        "origin_prediction_score",
        "origin_predicted_class",
        "origin_true_class_score",
        "embedding_start",
        "embedding_end",
        "label",
        "export_split",
    }
    missing = sorted(required_cols.difference(metadata_df.columns))
    if missing:
        raise RuntimeError(
            "Cannot build prototype bank because metadata is missing columns: "
            + ", ".join(missing)
        )

    if args.prototype_split == "train":
        candidate_df = metadata_df[metadata_df["export_split"] == "train"].copy()
    elif args.prototype_split == "train_val":
        candidate_df = metadata_df[metadata_df["export_split"].isin(["train", "val"])].copy()
    else:
        candidate_df = metadata_df.copy()

    if candidate_df.empty:
        raise RuntimeError(f"No rows available for prototype selection with split={args.prototype_split}.")
    candidate_df = _filter_prototype_normal_views(candidate_df, args)

    prototype_vectors = []
    bank_rows = []
    for class_idx in [0, 1]:
        class_df = candidate_df[candidate_df["label"].astype(int) == class_idx].copy()
        if class_df.empty:
            raise RuntimeError(f"No class {class_idx} rows available for prototype selection.")
        if class_idx == 0:
            class_df["selection_score"] = 1.0 - class_df["origin_prediction_score"].astype(float)
        else:
            class_df["selection_score"] = class_df["origin_prediction_score"].astype(float)
        class_df["selection_class"] = int(class_idx)

        ordered_positions = _prototype_sort_key(class_df)
        ordered_df = class_df.iloc[ordered_positions].reset_index(drop=True)
        if len(ordered_df) < args.prototype_k:
            print(
                f"[WARN] class {class_idx} has only {len(ordered_df)} candidates; "
                f"repeating rows to fill {args.prototype_k} prototypes."
            )

        for rank_idx in range(args.prototype_k):
            row = ordered_df.iloc[rank_idx % len(ordered_df)].to_dict()
            start = int(row["embedding_start"])
            end = int(row["embedding_end"])
            if end - start != 1:
                raise RuntimeError(
                    "bag_origin prototype selection expects exactly one embedding row per image; "
                    f"got {end - start} for sample_key={row.get('sample_key')}."
                )

            copied_path = _copy_prototype_image(row, args, class_idx, rank_idx)
            prototype_vectors.append(np.asarray(embeddings[start], dtype=np.float32))
            bank_rows.append(
                {
                    "head_name": "edl_head",
                    "prototype_global_idx": int(class_idx * args.prototype_k + rank_idx),
                    "prototype_class": int(class_idx),
                    "prototype_rank": int(rank_idx),
                    "prototype_id": f"c{class_idx}_p{rank_idx}",
                    "source_patient_id": str(row.get("patient_id", "")),
                    "source_image_id": str(row.get("image_id", "")),
                    "source_label": int(row["label"]),
                    "source_prediction_score": float(row["origin_prediction_score"]),
                    "source_true_class_score": float(row["origin_true_class_score"]),
                    "source_selection_score": float(row["selection_score"]),
                    "source_selection_class": int(row["selection_class"]),
                    "source_predicted_class": int(row["origin_predicted_class"]),
                    "source_correct": bool(int(row["origin_predicted_class"]) == int(row["label"])),
                    "source_split": str(row["export_split"]),
                    "source_sample_key": str(row.get("sample_key", "")),
                    "source_index": int(row.get("source_index", -1)),
                    "embedding_file": "embeddings.npy",
                    "embedding_start": start,
                    "embedding_end": end,
                    "embedding_row": start,
                    "image_path": str(row["image_path"]),
                    "prototype_image_path": str(copied_path),
                    "selection_rank": int(rank_idx),
                    "selection_method": "embedding_best_model",
                }
            )

    prototype_bank_df = pd.DataFrame(bank_rows)
    prototype_bank_path = args.out_dir / "prototype_bank.csv"
    prototype_bank_df.to_csv(prototype_bank_path, index=False)

    prototype_npz_path = args.out_dir / "prototype_bank.npz"
    prototype_array = np.asarray(prototype_vectors, dtype=np.float32).reshape(
        2,
        args.prototype_k,
        int(embeddings.shape[1]),
    )
    np.savez_compressed(
        prototype_npz_path,
        prototypes=prototype_array,
        prototype_class=prototype_bank_df["prototype_class"].to_numpy(dtype=np.int64),
        prototype_rank=prototype_bank_df["prototype_rank"].to_numpy(dtype=np.int64),
        embedding_row=prototype_bank_df["embedding_row"].to_numpy(dtype=np.int64),
        source_patient_id=prototype_bank_df["source_patient_id"].astype(str).to_numpy(),
        source_image_id=prototype_bank_df["source_image_id"].astype(str).to_numpy(),
        prototype_image_path=prototype_bank_df["prototype_image_path"].astype(str).to_numpy(),
    )
    return prototype_bank_path, prototype_npz_path, args.out_dir / "prototype_images"


def build_patch_metadata_rows(row, split_name, sample_key, coords, embedding_start, image_path):
    patient_id = str(row.get("patient_id", ""))
    image_id = str(row.get("image_id", ""))
    rows = []
    for patch_idx, coord in enumerate(coords.astype(int).tolist()):
        rows.append(
            {
                "export_split": split_name,
                "patient_id": patient_id,
                "image_id": image_id,
                "sample_key": sample_key,
                "patch_idx": patch_idx,
                "x": int(coord[0]),
                "y": int(coord[1]),
                "embedding_file": "embeddings.npy",
                "embedding_row": int(embedding_start + patch_idx),
                "image_path": str(image_path),
            }
        )
    return rows


def build_origin_patch_metadata_rows(row, split_name, sample_key, token_rows, embedding_start, image_path):
    patient_id = str(row.get("patient_id", ""))
    image_id = str(row.get("image_id", ""))
    rows = []
    for offset, token_row in enumerate(token_rows):
        rows.append(
            {
                "export_split": split_name,
                "patient_id": patient_id,
                "image_id": image_id,
                "sample_key": sample_key,
                "patch_idx": int(token_row["patch_idx"]),
                "token_idx": int(token_row["token_idx"]),
                "scale": int(token_row["scale"]),
                "cell_x": int(token_row["cell_x"]),
                "cell_y": int(token_row["cell_y"]),
                "x": int(token_row["x"]),
                "y": int(token_row["y"]),
                "token_width": int(token_row["token_width"]),
                "token_height": int(token_row["token_height"]),
                "embedding_file": "embeddings.npy",
                "embedding_row": int(embedding_start + offset),
                "image_path": str(image_path),
            }
        )
    return rows


def export_embeddings(args: argparse.Namespace):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = get_device(args)
    if args.gpu_id is not None:
        print(f"[INFO] Requested GPU id: {args.gpu_id}")
        print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] torch.cuda.current_device(): {torch.cuda.current_device()}")
    image_root = args.img_root if args.img_root is not None else args.data_dir / args.img_dir
    print(f"[INFO] Image root: {image_root}")

    split_dfs = load_and_split_dataframe(args)
    prepare_output_dir(args.out_dir, args.overwrite)

    image_encoder = None
    origin_model = None
    encoder_config = None
    image_encoder_load_msg = None
    origin_model_load_msg = None

    if args.embedding_level == "patch_encoder":
        image_encoder, encoder_config, image_encoder_load_msg = load_image_encoder(args, device)
        print(f"[INFO] Loaded image encoder from: {args.clip_chk_pt_path}")
        print(f"[INFO] Image encoder load message: {image_encoder_load_msg}")
    else:
        origin_model, origin_model_load_msg = load_origin_mil_model(args, device)
        print(f"[INFO] Loaded origin MIL checkpoint from: {args.origin_checkpoint}")
        print(f"[INFO] Origin model load message: {origin_model_load_msg}")

    embedding_chunks = []
    metadata_rows = []
    used_keys = set()
    split_counts = {}
    embedding_dims = set()
    embedding_offset = 0
    patch_preview_items = []
    patch_metadata_rows = []

    for split_name, split_df in split_dfs.items():
        split_counts[split_name] = int(len(split_df))
        dataset = FixedPatchBagDataset(args, split_df)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_single,
        )

        progress = tqdm(loader, desc=f"[{split_name}] export", total=len(loader))
        for sample in progress:
            patches = sample["x"]
            coords = sample["coords"]
            row = sample["row"]
            padding = sample["padding"]
            sample_key = make_sample_key(row, used_keys)
            original_num_patches = int(patches.shape[0])

            token_rows = None
            origin_prob = None
            if args.embedding_level == "patch_encoder":
                features = run_patch_encoder(image_encoder, patches, args, device)
                if int(features.shape[0]) != int(coords.shape[0]):
                    raise RuntimeError(
                        "Patch embedding row count does not match coordinate count: "
                        f"{features.shape[0]} embeddings vs {coords.shape[0]} coords "
                        f"for patient_id={row.get('patient_id')}, image_id={row.get('image_id')}."
                    )
                preview_coords = coords
            elif args.embedding_level == "origin_patch":
                features, token_rows = run_origin_patch(origin_model, patches, coords, args, device)
                preview_coords = np.asarray(
                    [[token_row["x"], token_row["y"]] for token_row in token_rows],
                    dtype=np.int32,
                )
            else:
                features, origin_prob = run_bag_origin(origin_model, patches, args, device)
                preview_coords = coords

            embedding_dims.add(int(features.shape[-1]))
            maybe_collect_patch_preview(
                patch_preview_items,
                args,
                row,
                split_name,
                sample_key,
                features,
                preview_coords,
                sample["image_path"],
            )
            if args.embedding_level == "patch_encoder":
                patch_metadata_rows.extend(
                    build_patch_metadata_rows(
                        row=row,
                        split_name=split_name,
                        sample_key=sample_key,
                        coords=coords,
                        embedding_start=embedding_offset,
                        image_path=sample["image_path"],
                    )
                )
            elif args.embedding_level == "origin_patch":
                patch_metadata_rows.extend(
                    build_origin_patch_metadata_rows(
                        row=row,
                        split_name=split_name,
                        sample_key=sample_key,
                        token_rows=token_rows,
                        embedding_start=embedding_offset,
                        image_path=sample["image_path"],
                    )
                )
            metadata_rows.append(
                build_metadata_row(
                    row=row,
                    split_name=split_name,
                    sample_key=sample_key,
                    features=features,
                    coords=coords,
                    args=args,
                    image_path=sample["image_path"],
                    padding=padding,
                    embedding_start=embedding_offset,
                    original_num_patches=original_num_patches,
                    origin_prob=origin_prob,
                )
            )
            embedding_chunks.append(features)
            embedding_offset += int(features.shape[0])

    if embedding_chunks:
        embeddings = np.concatenate(embedding_chunks, axis=0)
    else:
        dtype = np.float16 if args.dtype == "float16" else np.float32
        embeddings = np.empty((0, 0), dtype=dtype)

    embeddings_path = args.out_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = args.out_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    prototype_bank_path, prototype_npz_path, prototype_image_dir = save_embedding_prototype_bank(
        args,
        metadata_df,
        embeddings,
    )
    patch_metadata_path = None
    if patch_metadata_rows:
        patch_metadata_path = args.out_dir / "patch_metadata.csv"
        pd.DataFrame(patch_metadata_rows).to_csv(patch_metadata_path, index=False)
    preview_paths = save_patch_preview(args.out_dir, patch_preview_items)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command_args": {key: to_jsonable(value) for key, value in vars(args).items()},
        "clip_checkpoint": args.clip_chk_pt_path,
        "origin_checkpoint": args.origin_checkpoint,
        "embedding_level": args.embedding_level,
        "encoder_config": encoder_config,
        "image_encoder_load_msg": image_encoder_load_msg,
        "origin_model_load_msg": origin_model_load_msg,
        "split_counts": split_counts,
        "embedding_dims": sorted(embedding_dims),
        "embedding_file": str(embeddings_path),
        "metadata_file": str(metadata_path),
        "patch_metadata_file": str(patch_metadata_path) if patch_metadata_path else None,
        "patch_preview_file": str(preview_paths[0]) if preview_paths else None,
        "patch_preview_metadata_file": str(preview_paths[1]) if preview_paths else None,
        "prototype_k": int(args.prototype_k),
        "prototype_split": args.prototype_split,
        "prototype_bank_file": str(prototype_bank_path) if prototype_bank_path else None,
        "prototype_bank_npz_file": str(prototype_npz_path) if prototype_npz_path else None,
        "prototype_image_dir": str(prototype_image_dir) if prototype_image_dir else None,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "layout": {
            "embeddings.npy": "Flat array with shape [total_embedding_rows, embedding_dim].",
            "metadata.csv": (
                "One row per source image. Use embedding_start:embedding_end "
                "to slice that sample from embeddings.npy."
            ),
            "patch_metadata.csv": (
                "One row per exported patch/token for patch_encoder or origin_patch exports. "
                "embedding_row points to that vector in embeddings.npy."
            ),
            "prototype_bank.csv": (
                "For bag_origin exports with prototype_k > 0, one row per selected real-image "
                "prototype. embedding_row points to the prototype vector in embeddings.npy."
            ),
            "prototype_images/": "Copied source images grouped into negative/ and positive/ prototype folders.",
            "coords": "JSON list stored per metadata row; one [x, y] coordinate per original patch.",
        },
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as manifest_file:
        json.dump(to_jsonable(manifest), manifest_file, indent=2)

    print(f"[DONE] Embeddings saved to: {embeddings_path}")
    print(f"[DONE] Metadata saved to: {metadata_path}")
    if patch_metadata_path:
        print(f"[DONE] Patch metadata saved to: {patch_metadata_path}")
    if preview_paths:
        print(f"[DONE] Patch preview saved to: {preview_paths[0]}")
        print(f"[DONE] Patch preview metadata saved to: {preview_paths[1]}")
    if prototype_bank_path:
        print(f"[DONE] Prototype bank saved to: {prototype_bank_path}")
        print(f"[DONE] Prototype bank NPZ saved to: {prototype_npz_path}")
        print(f"[DONE] Prototype images saved under: {prototype_image_dir}")
    print(f"[DONE] Manifest saved to: {args.out_dir / 'manifest.json'}")
    for split_name in ["train", "val", "test"]:
        print(f"[DONE] {split_name}: {split_counts.get(split_name, 0)} samples")
    print(f"[DONE] Embedding shape: {list(embeddings.shape)}")
    print(f"[DONE] Embedding dims: {sorted(embedding_dims)}")


def main():
    args = normalize_args(parse_args())
    export_embeddings(args)


if __name__ == "__main__":
    main()
