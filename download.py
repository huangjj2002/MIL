import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import pandas as pd  # 用于处理数据
from tqdm import tqdm


# ========= 必改参数 =========
# 你的文件列表变量，请确保在运行此脚本前 mylist 已经定义
SOURCE_LIST = pd.read_csv('./new_csv.csv')['path_of_dicom'].tolist()  

BUCKET = "embed-dataset-open"
REGION = "us-west-2"

OUT_ROOT = Path("F:/new_dataset")  # 本地保存路径
MAX_WORKERS =12   # 并发数
OVERWRITE = False
# ===========================


class ProgressCallback:
    """用于 boto3 download_file 的进度回调"""
    def __init__(self, pbar):
        self.pbar = pbar

    def __call__(self, bytes_amount):
        self.pbar.update(bytes_amount)


def download_one(s3, bucket, key, out_root, overwrite=False):
    # key 在这里已经是正斜杠格式了，pathlib 会自动处理为本地路径
    dst = out_root / key
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return "skip", str(dst)

    try:
        # 先获取文件大小
        meta = s3.head_object(Bucket=bucket, Key=key)
        total_size = meta["ContentLength"]

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            desc=Path(key).name,
        ) as pbar:
            callback = ProgressCallback(pbar)
            tmp = dst.with_suffix(".part")
            s3.download_file(bucket, key, str(tmp), Callback=callback)
            os.replace(tmp, dst)

        return "ok", str(dst)
    except Exception as e:
        raise e


def main():
    # 1. 数据清洗：去空、去空白、去重、【关键】将反斜杠替换为正斜杠
    # 这样无论你的 list 里是 \ 还是 /，最终传给 S3 的都是标准的 /
    keys = [
        str(k).strip().replace("\\", "/") 
        for k in SOURCE_LIST 
        if pd.notna(k)
    ]
    keys = list(set(keys))
    
    print(f"Total unique files to download: {len(keys)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=REGION)

    ok = skip = fail = 0
    futures = []

    # 2. 提交下载任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for key in keys:
            # 这里不再强制检查 startswith("images/")，直接下载列表里的所有内容
            # 如果你确实想过滤，可以解开下面这行注释，并确保 key 用 / 格式
            # if not key.startswith("images/"):
            #     continue
            
            futures.append(ex.submit(
                download_one, s3, BUCKET, key, OUT_ROOT, OVERWRITE
            ))

        # 3. 等待完成并统计
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
            try:
                status, _ = fut.result()
                if status == "ok":
                    ok += 1
                else:
                    skip += 1
            except Exception as e:
                fail += 1
                # 打印错误方便调试，如果觉得太乱可以注释掉
                # print(f"\nError downloading: {e}")

    print("\n===== Download summary =====")
    print(f"Downloaded: {ok}")
    print(f"Skipped:    {skip}")
    print(f"Failed:     {fail}")
    print(f"Saved to:   {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
