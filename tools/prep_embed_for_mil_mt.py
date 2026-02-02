import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


# =========================
# 配置修改
# =========================
IN_CSV = "./new_csv.csv"  # 你的输入csv
OUT_ROOT = "F:/data"  # 输出根目录

IMG_DIR = "images_png"          
OUT_CSV_NAME = "mydata.csv"  


DICOM_COL = "path_of_dicom"
PATIENT_COL = "empi_anon"
LABEL_COL = "laterality_label" 


SUFFIX_MODE = "replace_ext" 
IMG_EXT = ".png"


SPLIT_VALUE = "training"

MAKE_TEST_SPLIT = False
TEST_FRAC = 0
TEST_RANDOM_SEED = 42


USE_WINDOW = True


SKIP_EXISTING = True


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
    """把 dicom 像素转成 0~255 uint8，尽量兼容乳腺片常见情况。"""
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
    """
    子进程/线程 worker：读取 dicom_path -> 写 out_png
    返回：(status, message)
      status: "ok" | "skipped" | "fail"
    """
    try:
        outp = Path(out_png)
        if skip_existing and outp.exists():
            return ("skipped", "exists")

        outp.parent.mkdir(parents=True, exist_ok=True)


        import pydicom
        import cv2

        ds = pydicom.dcmread(dicom_path, force=True)
        img_u8 = _dicom_to_uint8(ds, use_window=use_window)

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

    print(f"[1/4] 读取 CSV: {in_csv}")
    df = pd.read_csv(in_csv, low_memory=False)

    required = [DICOM_COL, PATIENT_COL, LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}\n当前列: {list(df.columns)}")


    if MAKE_TEST_SPLIT:
        print("[INFO] 生成 test split（按 patient 分组）")
        rng = np.random.RandomState(TEST_RANDOM_SEED)
        patients = df[PATIENT_COL].astype(str).unique().tolist()
        rng.shuffle(patients)
        n_test = int(round(len(patients) * TEST_FRAC))
        test_set = set(patients[:n_test])
        df["_split"] = df[PATIENT_COL].astype(str).map(lambda x: "test" if x in test_set else "training")
    else:
        df["_split"] = SPLIT_VALUE

    print(f"[2/4] 构建任务列表（png 输出到: {out_img_root}）")
    tasks = []
    records = []
    for i, row in df.iterrows():
        dicom_path = str(row[DICOM_COL])
        patient_id = str(row[PATIENT_COL])
        image_id = build_image_id_from_dicom_path(dicom_path) 
        out_png = out_img_root / patient_id / image_id

        cancer = label_to_cancer(row[LABEL_COL])
        records.append({
            "patient_id": patient_id,
            "image_id": image_id,
            "split": row["_split"],
            "cancer": int(cancer),
        })

        tasks.append((i, dicom_path, str(out_png)))


    print(f"[3/4] 开始并行转换：workers={MAX_WORKERS} mode={'process' if USE_MULTIPROCESS else 'thread'}")
    ok = skipped = fail = 0
    fail_samples = []

    Executor = ProcessPoolExecutor if USE_MULTIPROCESS else ThreadPoolExecutor
    with Executor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_convert_one, dicom_path, out_png, USE_WINDOW, SKIP_EXISTING): (idx, dicom_path, out_png)
            for (idx, dicom_path, out_png) in tasks
        }

        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, dicom_path, out_png = futures[fut]
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                fail += 1
                fail_samples.append((idx, dicom_path, out_png, msg))


    print(f"[4/4] 写出项目 CSV: {out_csv_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_csv_path, index=False)

    print("==== 完成 ====")
    print(f"PNG 输出目录: {out_img_root}")
    print(f"项目 CSV: {out_csv_path}")
    print(f"转换成功: {ok}, 跳过已存在: {skipped}, 失败: {fail}")

    if fail_samples:
        fail_csv = out_root / "convert_failures.csv"
        pd.DataFrame(fail_samples, columns=["row_index", "dicom_path", "out_png", "error"]).to_csv(fail_csv, index=False)
        print(f"[WARN] 有失败样本，已写出失败清单: {fail_csv}")
        for r in fail_samples[:10]:
            print(f"[FAIL] row={r[0]} dicom={r[1]} -> {r[2]} err={r[3]}")

    print("\n训练提示（对应你原项目 online）：")
    print(f"  --data_dir {OUT_ROOT}")
    print(f"  --img_dir {IMG_DIR}")
    print(f"  --csv_file {OUT_CSV_NAME}")
    print(f"  label 列名为：cancer")


if __name__ == "__main__":
    main()
