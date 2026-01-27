import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 配置区域 =================
INPUT_LIST_CSV = 'list_for_conversion_debug.csv' # csv文件地址
# 路径
EMBED_ROOT = r"."  
OUTPUT_DIR = "embed_data"
FINAL_CSV_NAME = 'embed_data.csv'

# 进程数
NUM_WORKERS = max(1, os.cpu_count() - 2) 

# [新增] 强制 Resize 尺寸 (Width, Height)
# 注意：cv2.resize 的参数顺序是 (宽, 高)。
# 对应 main.py 中的默认 img_size=[1520, 912] (通常是 H, W)，这里设置为 (912, 1520)
TARGET_SIZE = (912, 1520) 
# ===========================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def convert_dicom_to_png(dicom_path, output_path):

    try:
        dcm = pydicom.dcmread(dicom_path)
        
        try:
            image = apply_voi_lut(dcm.pixel_array, dcm)
        except:
            image = dcm.pixel_array.astype(float)
            

        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
            image = np.amax(image) - image
        

        image = image.astype(np.float32)
        img_min = image.min()
        img_max = image.max()
        
        if img_max - img_min != 0:
            image = (image - img_min) / (img_max - img_min) * 255.0
        else:
            image = np.zeros_like(image)
            
        image = image.astype(np.uint8)

        # [新增] Resize 逻辑
        if TARGET_SIZE is not None:
            # cv2.resize 接受 (Width, Height)
            image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        # print(f"Error converting {dicom_path}: {e}") # 可选：打印错误信息帮助调试
        return False

def process_single_row(row_data, embed_root, images_out_root):

    patient_id = str(row_data['patient_id'])
    image_id = str(row_data['image_id'])
    dicom_rel_path = str(row_data['dicom_path'])
    

    clean_rel_path = dicom_rel_path.strip("/").strip("\\")
    dicom_full_path = os.path.join(embed_root, clean_rel_path)
    

    if not os.path.exists(dicom_full_path):
        dicom_full_path = dicom_full_path.replace("\\", "/")
        
 
    patient_dir = os.path.join(images_out_root, patient_id)

    os.makedirs(patient_dir, exist_ok=True)
    
    target_path = os.path.join(patient_dir, image_id) # 注意：这里可能需要加后缀 .png，或者 output_path 已经包含后缀
    # 根据原代码逻辑，image_id 似乎没有后缀，OpenCV 保存需要后缀名才能正确编码，
    # 但原代码直接用 image_id 作为文件名，且 cv2.imwrite(target_path, image)。
    # 如果 image_id 里没有 .png，cv2可能会报错或者不知道存什么格式。
    # 建议加上后缀，但为了保持"其他地方不变"，这里暂按原样，除非 image_id 本身带后缀。
    # 稳妥起见，建议在 target_path 强制加 .png，如下行所示（如果 image_id 已经带了后缀则会重复，请自行确认）：
    if not target_path.lower().endswith('.png'):
        target_path += ".png"

    success = False
    
    # 检查目标文件是否存在且大小大于0
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        success = True
    
    elif os.path.exists(dicom_full_path):
        success = convert_dicom_to_png(dicom_full_path, target_path)
    

    if success:
        if 'dicom_path' in row_data:
            del row_data['dicom_path']
        
        # [可选] 更新 row_data 中的 filename 为新的 png 文件名
        # row_data['image_file_path'] = target_path 
        
        return row_data
    else:
        return None

def main():
    if not os.path.exists(INPUT_LIST_CSV):
        print(f"错误：找不到输入文件 {INPUT_LIST_CSV}")
        return

    df = pd.read_csv(INPUT_LIST_CSV)
    
    images_out_root = os.path.join(OUTPUT_DIR, "images_png")
    ensure_dir(images_out_root)
    
    print(f">>> Step 2: 开始转换 {len(df)} 张图像...")
    print(f"    使用核心数: {NUM_WORKERS}")
    print(f"    源目录: {EMBED_ROOT}")
    print(f"    目标尺寸: {TARGET_SIZE}")
    
    valid_records = []
    

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        futures = [
            executor.submit(process_single_row, row.to_dict(), EMBED_ROOT, images_out_root) 
            for _, row in df.iterrows()
        ]
        
  
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                valid_records.append(result)
    

    print("\n" + "="*30)
    print(f"处理完成！成功转换: {len(valid_records)} / {len(df)}")
    
    if len(valid_records) > 0:
        final_df = pd.DataFrame(valid_records)
        final_csv_path = os.path.join(OUTPUT_DIR, FINAL_CSV_NAME)
        final_df.to_csv(final_csv_path, index=False)
        print(f">>> 最终 CSV 已保存: {final_csv_path}")
    else:
        print(">>> 未生成任何有效数据，请检查路径设置。")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()