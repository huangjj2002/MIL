"""Export reusable origin/Mammo-CLIP encoder embeddings.

The default output is patch-level image-encoder embeddings stored as one
continuous embeddings.npy array, plus metadata.csv and manifest.json. These
features are intended as a fast input cache for later EDL, DST, or
prototype-style modules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


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
    parser.add_argument("--data-dir", default="datasets/Vindir-mammoclip", type=str)
    parser.add_argument(
        "--img-dir",
        default="VinDir_preprocessed_mammoclip/images_png",
        type=str,
        help="Image directory relative to data-dir.",
    )
    parser.add_argument("--csv-file", default="grouped_df.csv", type=str)
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

    # Embedding level
    parser.add_argument(
        "--embedding-level",
        "--embedding_level",
        dest="embedding_level",
        default="patch_encoder",
        choices=["patch_encoder", "bag_origin"],
        help="patch_encoder exports Mammo-CLIP image-encoder patch embeddings; bag_origin exports origin MIL classifier-input embeddings.",
    )
    parser.add_argument(
        "--origin-checkpoint",
        "--origin_checkpoint",
        dest="origin_checkpoint",
        default=None,
        type=str,
        help="Origin MIL best_model.pth; required for --embedding-level bag_origin.",
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
    if args.embedding_level == "bag_origin" and args.origin_checkpoint is None:
        raise ValueError("--origin-checkpoint is required when --embedding-level bag_origin.")
    if args.embedding_level == "bag_origin" and args.multi_scale_model == "msp":
        raise ValueError("bag_origin export does not support multi_scale_model=msp in this script.")

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
        self.img_root = args.data_dir / args.img_dir
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

    csv_path = args.data_dir / args.csv_file
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
                _ = model(inputs)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Could not capture bag_origin embedding from model.classifier input.")
    bag_embedding = captured[-1].reshape(1, -1)
    target_dtype = np.float16 if args.dtype == "float16" else np.float32
    return bag_embedding.numpy().astype(target_dtype, copy=False)


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
    return metadata_row


def export_embeddings(args: argparse.Namespace):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = get_device(args)
    print(f"[INFO] Using device: {device}")

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

            if args.embedding_level == "patch_encoder":
                features = run_patch_encoder(image_encoder, patches, args, device)
            else:
                features = run_bag_origin(origin_model, patches, args, device)

            embedding_dims.add(int(features.shape[-1]))
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
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "layout": {
            "embeddings.npy": "Flat array with shape [total_embedding_rows, embedding_dim].",
            "metadata.csv": (
                "One row per source image. Use embedding_start:embedding_end "
                "to slice that sample from embeddings.npy."
            ),
            "coords": "JSON list stored per metadata row; one [x, y] coordinate per original patch.",
        },
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as manifest_file:
        json.dump(to_jsonable(manifest), manifest_file, indent=2)

    print(f"[DONE] Embeddings saved to: {embeddings_path}")
    print(f"[DONE] Metadata saved to: {metadata_path}")
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
