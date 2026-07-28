"""Export patch/token uncertainty heatmaps from a trained bag-embedding DST-Prototype head.

This script uses a trained DST-Prototype checkpoint whose head was trained on
bag_origin embeddings, then applies that same prototype head to origin_patch
embeddings from the same origin MIL model. The result is an interpretable,
approximate patch/token uncertainty map, not a retrained patch-level DST model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from MIL.edl_proto_models import BagEmbeddingPrototypeDSTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export DST-Prototype uncertainty heatmaps.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Trained DST_PROTO best_model.pth.")
    parser.add_argument(
        "--patch-cache-dir",
        "--patch_cache_dir",
        dest="patch_cache_dir",
        required=True,
        type=str,
        help="origin_patch cache directory with embeddings.npy, metadata.csv, patch_metadata.csv.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None, type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--scale",
        default=None,
        type=int,
        help="FPN scale to visualize. If omitted, the smallest available scale is used.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", default=4096, type=int)
    parser.add_argument("--max-images", "--max_images", dest="max_images", default=None, type=int)
    parser.add_argument("--heatmap-alpha", "--heatmap_alpha", dest="heatmap_alpha", default=0.45, type=float)
    parser.add_argument("--colormap", default="magma", type=str)
    parser.add_argument(
        "--normalize-heatmap",
        "--normalize_heatmap",
        dest="normalize_heatmap",
        default="y",
        choices=["y", "n"],
        help="Normalize each rendered heatmap inside its covered region for visibility.",
    )
    parser.add_argument(
        "--selected-csv",
        "--selected_csv",
        dest="selected_csv",
        default=None,
        type=str,
        help=(
            "Optional selected-case CSV. If it has patient_id and image_id, those are used "
            "directly. If it only has short DICOM names like TP_01.dcm, also pass "
            "--selected-dcm-root so SOPInstanceUID can be read from the DICOM headers."
        ),
    )
    parser.add_argument(
        "--selected-dcm-root",
        "--selected_dcm_root",
        dest="selected_dcm_root",
        default=None,
        type=str,
        help=(
            "Optional root containing selected DICOMs, for example a folder with "
            "TPcase/FPcase/TNcase subfolders from the review bundle."
        ),
    )
    parser.add_argument("--edl-proto-normalize", "--edl_proto_normalize", dest="edl_proto_normalize",
                        default="y", choices=["y", "n"])
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", default=None, type=str)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-patch-csv", "--save_patch_csv", dest="save_patch_csv",
                        default="y", choices=["y", "n"])
    return parser.parse_args()


def sanitize_key(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def get_device(args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    return checkpoint, state


def infer_head_shape(state):
    weight = state.get("edl_head.ds_module.ds1.w")
    if weight is None:
        raise ValueError(
            "Expected a bag-embedding DST_PROTO checkpoint with edl_head.ds_module.ds1.w."
        )
    n_prototypes, in_features = tuple(weight.shape)
    if n_prototypes % 2 != 0:
        raise ValueError(f"Expected binary prototype count divisible by 2, got {n_prototypes}.")
    return int(n_prototypes // 2), int(in_features)


def load_model(args, device, patch_dim):
    checkpoint, state = load_checkpoint_state(Path(args.checkpoint))
    proto_k, in_features = infer_head_shape(state)
    if int(patch_dim) != int(in_features):
        raise ValueError(
            "Patch embedding dimension does not match the trained DST head: "
            f"patch dim={patch_dim}, checkpoint dim={in_features}. "
            "Use origin_patch embeddings extracted with the same origin model/config."
        )

    model = BagEmbeddingPrototypeDSTModel(
        in_features=in_features,
        edl_dropout=0.0,
        proto_k=proto_k,
        proto_topk=1,
        proto_normalize=args.edl_proto_normalize == "y",
    )
    load_msg = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    model.is_training = False
    return model, checkpoint, str(load_msg), proto_k


@torch.no_grad()
def predict_patch_scores(model, vectors, device, batch_size):
    chunks = {
        "prediction_score": [],
        "predicted_class": [],
        "uncertainty": [],
        "mass_0": [],
        "mass_1": [],
        "mass_omega": [],
    }
    for start in range(0, vectors.shape[0], batch_size):
        batch = torch.as_tensor(
            np.asarray(vectors[start:start + batch_size], dtype=np.float32),
            device=device,
        )
        out = model(batch)
        prob = out["prob"].detach().cpu().numpy()
        mass = out["dst_mass"].detach().cpu().numpy()
        uncertainty = out["uncertainty"].detach().cpu().numpy()
        chunks["prediction_score"].append(prob[:, 1])
        chunks["predicted_class"].append(np.argmax(prob, axis=1))
        chunks["uncertainty"].append(uncertainty)
        chunks["mass_0"].append(mass[:, 0])
        chunks["mass_1"].append(mass[:, 1])
        chunks["mass_omega"].append(mass[:, 2])
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    embeddings_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "metadata.csv"
    patch_metadata_path = cache_dir / "patch_metadata.csv"
    manifest_path = cache_dir / "manifest.json"
    for path in [embeddings_path, metadata_path, patch_metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required origin_patch cache file missing: {path}")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path).fillna(0)
    patch_metadata = pd.read_csv(patch_metadata_path).fillna(0)
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("embedding_level") != "origin_patch":
            print(
                f"[WARN] Expected origin_patch cache, got embedding_level={manifest.get('embedding_level')!r}."
            )
    return embeddings, metadata, patch_metadata, manifest


def choose_scale(patch_metadata, requested_scale):
    if "scale" not in patch_metadata.columns:
        return None
    scales = sorted(int(scale) for scale in patch_metadata["scale"].dropna().unique())
    if not scales:
        return None
    if requested_scale is not None:
        if int(requested_scale) not in scales:
            raise ValueError(f"Requested --scale {requested_scale} not found. Available scales: {scales}")
        return int(requested_scale)
    selected = int(scales[0])
    print(f"[INFO] --scale not provided; using finest available scale: {selected}")
    return selected


def metadata_lookup(metadata):
    lookup = {}
    for _, row in metadata.iterrows():
        key = make_lookup_key(row.get("patient_id", ""), row.get("image_id", ""))
        lookup[key] = row.to_dict()
    return lookup


def normalize_key_part(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_image_id(value):
    text = normalize_key_part(value)
    if not text:
        return text
    if text.lower().endswith(".dcm"):
        text = text[:-4] + ".png"
    elif "." not in Path(text).name:
        text = f"{text}.png"
    return text


def make_lookup_key(patient_id, image_id):
    return normalize_key_part(patient_id), normalize_image_id(image_id)


def selected_case_folder(image_name):
    prefix = str(image_name).split("_", 1)[0].upper()
    return {
        "TP": "TPcase",
        "TN": "TNcase",
        "FP": "FPcase",
    }.get(prefix)


def resolve_selected_dicom(dcm_root, image_name):
    root = Path(dcm_root)
    folder = selected_case_folder(image_name)
    candidates = [root / image_name]
    if folder:
        candidates.extend([
            root / folder / image_name,
            root / "data" / folder / image_name,
        ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve selected DICOM {image_name!r} under {dcm_root}. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def read_dicom_lookup_values(dcm_path):
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required when --selected-csv lacks original image_id and "
            "--selected-dcm-root is used."
        ) from exc

    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    patient_id = normalize_key_part(getattr(ds, "PatientID", ""))
    sop_uid = normalize_key_part(getattr(ds, "SOPInstanceUID", ""))
    if not sop_uid:
        raise ValueError(f"SOPInstanceUID missing from selected DICOM: {dcm_path}")
    return patient_id, f"{sop_uid}.png"


def load_selected_lookup(selected_csv, selected_dcm_root=None):
    if not selected_csv:
        return {}

    selected_df = pd.read_csv(selected_csv).fillna("")
    lookup = {}
    for _, row in selected_df.iterrows():
        image_name = normalize_key_part(row.get("image_name", ""))
        group = normalize_key_part(row.get("group", ""))

        image_id = ""
        for col in ["image_id", "source_image_id", "original_image_id"]:
            if col in selected_df.columns and normalize_key_part(row.get(col, "")):
                image_id = normalize_image_id(row.get(col, ""))
                break

        patient_id = normalize_key_part(row.get("patient_id", ""))
        if not image_id:
            if not selected_dcm_root:
                raise ValueError(
                    f"Selected row for {image_name or '<unknown>'} has no image_id. "
                    "Pass --selected-dcm-root to recover SOPInstanceUID from DICOM headers."
                )
            if not image_name:
                raise ValueError("Selected CSV row has no image_name and no image_id.")
            dcm_path = resolve_selected_dicom(selected_dcm_root, image_name)
            dcm_patient_id, image_id = read_dicom_lookup_values(dcm_path)
            patient_id = patient_id or dcm_patient_id

        key = make_lookup_key(patient_id, image_id)
        lookup[key] = {
            "selection_group": group,
            "selection_image_name": image_name,
        }

    print(f"[INFO] Loaded selected-case filter: {len(lookup)} images from {selected_csv}")
    return lookup


def render_heatmap(image_path, rows, scores, meta_row, output_path, alpha, colormap_name, normalize_heatmap):
    from PIL import Image
    import matplotlib

    image = Image.open(image_path).convert("RGB")
    image_arr = np.asarray(image).astype(np.float32)
    height, width = image_arr.shape[:2]

    heat_sum = np.zeros((height, width), dtype=np.float32)
    heat_count = np.zeros((height, width), dtype=np.float32)
    padding_left = int(float(meta_row.get("padding_left", 0))) if meta_row is not None else 0
    padding_top = int(float(meta_row.get("padding_top", 0))) if meta_row is not None else 0

    default_width = int(rows["token_width"].median()) if "token_width" in rows.columns else 32
    default_height = int(rows["token_height"].median()) if "token_height" in rows.columns else default_width

    for (_, row), value in zip(rows.iterrows(), scores):
        token_width = int(row.get("token_width", default_width))
        token_height = int(row.get("token_height", default_height))
        x0 = int(row["x"]) - padding_left
        y0 = int(row["y"]) - padding_top
        x1 = x0 + token_width
        y1 = y0 + token_height
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        heat_sum[y0:y1, x0:x1] += float(value)
        heat_count[y0:y1, x0:x1] += 1.0

    mask = heat_count > 0
    heat = np.zeros_like(heat_sum)
    heat[mask] = heat_sum[mask] / heat_count[mask]
    if normalize_heatmap and mask.any():
        covered = heat[mask]
        min_value = float(covered.min())
        max_value = float(covered.max())
        if max_value > min_value:
            heat[mask] = (covered - min_value) / (max_value - min_value)
        else:
            heat[mask] = 0.0
    heat = np.clip(heat, 0.0, 1.0)

    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    color = np.asarray(cmap(heat)[..., :3], dtype=np.float32) * 255.0
    alpha_map = (alpha * mask.astype(np.float32))[..., None]
    overlay = image_arr * (1.0 - alpha_map) + color * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return output_path


def export_heatmaps(args):
    device = get_device(args)
    print(f"[INFO] Using device: {device}")

    embeddings, metadata, patch_metadata, manifest = load_cache(args.patch_cache_dir)
    selected_scale = choose_scale(patch_metadata, args.scale)
    if selected_scale is not None:
        patch_metadata = patch_metadata[patch_metadata["scale"].astype(int) == selected_scale].reset_index(drop=True)

    if args.split != "all":
        patch_metadata = patch_metadata[patch_metadata["export_split"].astype(str) == args.split].reset_index(drop=True)
    if patch_metadata.empty:
        raise RuntimeError("No patch metadata rows left after split/scale filtering.")

    selected_lookup = load_selected_lookup(args.selected_csv, args.selected_dcm_root)
    if selected_lookup:
        patch_keys = [
            make_lookup_key(patient_id, image_id)
            for patient_id, image_id in zip(patch_metadata["patient_id"], patch_metadata["image_id"])
        ]
        keep_mask = [key in selected_lookup for key in patch_keys]
        patch_metadata = patch_metadata.loc[keep_mask].reset_index(drop=True)
        if patch_metadata.empty:
            raise RuntimeError("No patch metadata rows matched the selected-case filter.")
        print(f"[INFO] Selected-case patch rows kept: {len(patch_metadata)}")

    model, checkpoint, load_msg, proto_k = load_model(args, device, embeddings.shape[1])
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    print(f"[INFO] Model load message: {load_msg}")
    print(f"[INFO] Inferred DST prototype k: {proto_k}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "dst_proto_uncertainty_heatmaps"
    heatmap_dir = output_dir / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_lookup = metadata_lookup(metadata)
    score_rows = []
    groups = list(patch_metadata.groupby(["patient_id", "image_id"], sort=False))
    if args.max_images is not None:
        groups = groups[: int(args.max_images)]

    for group_index, ((patient_id, image_id), rows) in enumerate(groups, start=1):
        rows = rows.reset_index(drop=True)
        selected_info = selected_lookup.get(make_lookup_key(patient_id, image_id), {})
        embedding_rows = rows["embedding_row"].astype(int).to_numpy()
        vectors = embeddings[embedding_rows]
        scores = predict_patch_scores(model, vectors, device, args.batch_size)

        out_rows = rows.copy()
        for key, value in scores.items():
            out_rows[key] = value
        if selected_info:
            out_rows["selection_group"] = selected_info.get("selection_group", "")
            out_rows["selection_image_name"] = selected_info.get("selection_image_name", "")
        score_rows.append(out_rows)

        meta_row = meta_lookup.get(make_lookup_key(patient_id, image_id), None)
        image_path = (
            str(meta_row.get("image_path"))
            if meta_row is not None and meta_row.get("image_path")
            else str(rows["image_path"].iloc[0])
        )
        name_prefix = ""
        if selected_info:
            group_name = sanitize_key(selected_info.get("selection_group", ""))
            selected_name = sanitize_key(Path(selected_info.get("selection_image_name", "")).stem)
            name_prefix = "_".join(part for part in [group_name, selected_name] if part)
            if name_prefix:
                name_prefix = f"{name_prefix}_"
        filename = (
            f"{name_prefix}{group_index:05d}_p-{sanitize_key(patient_id)}_"
            f"i-{sanitize_key(image_id)}_uncertainty.png"
        )
        heatmap_output_dir = heatmap_dir
        if selected_info and selected_info.get("selection_group"):
            heatmap_output_dir = heatmap_dir / sanitize_key(selected_info["selection_group"])
        render_heatmap(
            image_path=image_path,
            rows=rows,
            scores=scores["uncertainty"],
            meta_row=meta_row,
            output_path=heatmap_output_dir / filename,
            alpha=float(args.heatmap_alpha),
            colormap_name=args.colormap,
            normalize_heatmap=args.normalize_heatmap == "y",
        )

    patch_scores_path = None
    if args.save_patch_csv == "y" and score_rows:
        patch_scores = pd.concat(score_rows, ignore_index=True)
        patch_scores_path = output_dir / "dst_proto_uncertainty_patch_scores.csv"
        patch_scores.to_csv(patch_scores_path, index=False)

    manifest_out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(Path(args.checkpoint)),
        "patch_cache_dir": str(Path(args.patch_cache_dir)),
        "cache_embedding_level": manifest.get("embedding_level"),
        "split": args.split,
        "scale": selected_scale,
        "proto_k": proto_k,
        "heatmap_dir": str(heatmap_dir),
        "patch_scores_csv": str(patch_scores_path) if patch_scores_path else None,
        "num_images": len(groups),
        "selected_csv": str(Path(args.selected_csv)) if args.selected_csv else None,
        "selected_dcm_root": str(Path(args.selected_dcm_root)) if args.selected_dcm_root else None,
        "selected_images": len(selected_lookup),
        "normalize_heatmap": args.normalize_heatmap == "y",
        "note": (
            "Patch/token uncertainty is computed by applying the trained bag-level "
            "DST prototype head to origin_patch embeddings."
        ),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"[DONE] Heatmaps saved under: {heatmap_dir}")
    if patch_scores_path:
        print(f"[DONE] Patch uncertainty CSV saved: {patch_scores_path}")
    print(f"[DONE] Manifest saved: {output_dir / 'manifest.json'}")


def main():
    args = parse_args()
    export_heatmaps(args)


if __name__ == "__main__":
    main()
