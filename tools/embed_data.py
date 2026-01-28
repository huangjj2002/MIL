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


        if TARGET_SIZE is not None:
    
            image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
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
    
    target_path = os.path.join(patient_dir, image_id) 

    if not target_path.lower().endswith('.png'):
        target_path += ".png"

    success = False
    

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        success = True
    
    elif os.path.exists(dicom_full_path):
        success = convert_dicom_to_png(dicom_full_path, target_path)
    

    if success:
        if 'dicom_path' in row_data:
            del row_data['dicom_path']
        

        
        return row_data
    else:
        return None

def main():
    if not os.path.exists(INPUT_LIST_CSV):
        print(f"错误：找不到csv文件 {INPUT_LIST_CSV}")
        return

    df = pd.read_csv(INPUT_LIST_CSV)
    
    images_out_root = os.path.join(OUTPUT_DIR, "images_png")
    ensure_dir(images_out_root)
    
    print(f"    转换 {len(df)} 张图像...")
    print(f"    进程数: {NUM_WORKERS}")
    print(f"    源目录地址: {EMBED_ROOT}")
    print(f"    目标图像尺寸: {TARGET_SIZE}")
    
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
        print(f"CSV 已保存: {final_csv_path}")
    else:
        print("未找到数据，请检查路径设置。")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()