import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm



IN_CSV = "./new_csv.csv"#被筛选出的implant的csv文件地址 
OUT_ROOT = "F:/data"  #输出地址

IMG_DIR = "images_png" #png的存储路径         
OUT_CSV_NAME = "embed_data.csv" #用于模型输入的csv文件
#图片的存储路径为OUT_ROOT/IMG_DIR/empi_anon/.png

DICOM_COL = "anon_dicom_path"#IN_CSV文件中记录dicom路径的列名
PATIENT_COL = "empi_anon"
LABEL_COL = "laterality_label"  #用以指定标签名


SUFFIX_MODE = "replace_ext" 
IMG_EXT = ".png"#文件后缀名


SPLIT_VALUE = "training"

MAKE_TEST_SPLIT = False
TEST_FRAC = 0
TEST_RANDOM_SEED = 42


USE_WINDOW = True

TARGET_SIZE_HW = (1520, 912)  
INTERPOLATION = "area"       

SKIP_EXISTING = True

# 并行配置
USE_MULTIPROCESS = True  
MAX_WORKERS = max(1, min(16, (os.cpu_count() or 8))) 




def _to_float(x):

    if isinstance(x, (list, tuple)):
        return float(x[0])
    try:
        return float(x)
    except Exception:
        return None


def build_image_id_from_dicom_path(dicom_path: str) -> str:

    base = os.path.basename(str(dicom_path))  
    if SUFFIX_MODE == "append_ext":
        return base + IMG_EXT 
    stem, _ = os.path.splitext(base)         
    return stem + IMG_EXT                     


def label_to_cancer(v) -> int:

    try:
        fv = float(v)
        return 1 if fv > 0 else 0
    except Exception:
        s = str(v).strip().lower()
        if s in ("", "0", "false", "no", "neg", "negative", "none", "nan"):
            return 0
        return 1


def _dicom_to_uint8(ds, use_window: bool = True) -> np.ndarray:

    arr = ds.pixel_array.astype(np.float32)


    photo = getattr(ds, "PhotometricInterpretation", "")
    if photo == "MONOCHROME1":
        arr = arr.max() - arr


    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    if use_window:
        wc = _to_float(getattr(ds, "WindowCenter", None))
        ww = _to_float(getattr(ds, "WindowWidth", None))
        if wc is not None and ww is not None and ww > 0:
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
            arr = np.clip(arr, lo, hi)
        else:
            lo = np.percentile(arr, 1)
            hi = np.percentile(arr, 99)
            if hi > lo:
                arr = np.clip(arr, lo, hi)


    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def _convert_one(dicom_path: str, out_png: str, use_window: bool, skip_existing: bool) -> tuple:

    try:
        outp = Path(out_png)
        if skip_existing and outp.exists():
            return ("skipped", "exists")

        outp.parent.mkdir(parents=True, exist_ok=True)


        import pydicom
        import cv2

        ds = pydicom.dcmread(dicom_path, force=True)
        img_u8 = _dicom_to_uint8(ds, use_window=use_window)


        target_h, target_w = TARGET_SIZE_HW
        interp = cv2.INTER_AREA if INTERPOLATION == "area" else cv2.INTER_LINEAR
        if img_u8.shape[0] != target_h or img_u8.shape[1] != target_w:
            img_u8 = cv2.resize(img_u8, (target_w, target_h), interpolation=interp)

        ok = cv2.imwrite(str(outp), img_u8)
        if not ok:
            return ("fail", "cv2.imwrite failed")
        return ("ok", "")
    except Exception as e:
        return ("fail", repr(e))


def main():
    in_csv = Path(IN_CSV)
    out_root = Path(OUT_ROOT)
    out_img_root = out_root / IMG_DIR
    out_csv_path = out_root / OUT_CSV_NAME

    if not in_csv.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {in_csv}")

    out_img_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"读取 CSV: {in_csv}")
    df = pd.read_csv(in_csv, low_memory=False)

    required = [DICOM_COL, PATIENT_COL, LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}\n当前列: {list(df.columns)}")


    if MAKE_TEST_SPLIT:
        rng = np.random.RandomState(TEST_RANDOM_SEED)
        patients = df[PATIENT_COL].astype(str).unique().tolist()
        rng.shuffle(patients)
        n_test = int(round(len(patients) * TEST_FRAC))
        test_set = set(patients[:n_test])
        df["_split"] = df[PATIENT_COL].astype(str).map(lambda x: "test" if x in test_set else "training")
    else:
        df["_split"] = SPLIT_VALUE

    records = []
    unique_map = {}

    for _, row in df.iterrows():
        dicom_path = str(row[DICOM_COL])
        patient_id = str(row[PATIENT_COL])
        image_id = build_image_id_from_dicom_path(dicom_path) 
        out_png = str(out_img_root / patient_id / image_id)

        cancer = label_to_cancer(row[LABEL_COL])
        records.append({
            "patient_id": patient_id,
            "image_id": image_id,
            "split": row["_split"],
            "cancer": int(cancer),
        })


        if out_png not in unique_map:
            unique_map[out_png] = dicom_path

    total_rows = len(df)
    total_unique_tasks = len(unique_map)
    dedup_reduced = total_rows - total_unique_tasks


    ok = skipped = fail = 0
    fail_samples = []

    Executor = ProcessPoolExecutor if USE_MULTIPROCESS else ThreadPoolExecutor
    with Executor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_convert_one, dicom_path, out_png, USE_WINDOW, SKIP_EXISTING): (out_png, dicom_path)
            for out_png, dicom_path in unique_map.items()
        }

        for fut in tqdm(as_completed(futures), total=len(futures)):
            out_png, dicom_path = futures[fut]
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                fail += 1
                fail_samples.append((dicom_path, out_png, msg))

    print(f"输出CSV: {out_csv_path}")
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_csv_path, index=False)

    print(f"PNG 输出目录: {out_img_root}")
    print(f"转换成功: {ok}, 跳过已存在: {skipped}, 失败: {fail}")

    if fail_samples:
        fail_csv = out_root / "convert_failures.csv"
        pd.DataFrame(fail_samples, columns=["dicom_path", "out_png", "error"]).to_csv(fail_csv, index=False)
        print(f"[WARN] 有失败样本，已输出失败log: {fail_csv}")
        for r in fail_samples[:10]:
            print(f"[FAIL] dicom={r[0]} -> {r[1]} err={r[2]}")

    print("\n")
    print(f"  --data_dir {OUT_ROOT}")
    print(f"  --img_dir {IMG_DIR}")
    print(f"  --csv_file {OUT_CSV_NAME}")
    print("  label 列名为：cancer")


if __name__ == "__main__":
    main()
