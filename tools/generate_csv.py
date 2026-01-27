import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
CLINICAL_FILE = '../EMBED_OpenData_clinical.csv'
METADATA_FILE = '../EMBED_OpenData_metadata.csv'
OUTPUT_CSV_FOR_CONVERSION = 'list_for_conversion.csv' # 给脚本二用的中间文件
# ===========================================

def main():
    print(">>> Step 1: 读取数据...")
    df_clin = pd.read_csv(CLINICAL_FILE, low_memory=False)
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)

    print(f"原始记录 -> 临床: {len(df_clin)} | 元数据: {len(df_meta)}")

    # -------------------------------------------------------
    # 1. 图像层级筛选 (Metadata)
    # -------------------------------------------------------
    # 去除植入物
    if 'BreastImplantPresent' in df_meta.columns:
        df_meta = df_meta[df_meta['BreastImplantPresent'] != 'YES']
    
    # 只保留 2D 图像 (根据需要调整，标准 MIL 仅支持 2D)
    if 'FinalImageType' in df_meta.columns:
        # 如果您想包含 C-View，可以把下面改为: .isin(['2D', 'GENERATED_2D'])
        df_meta = df_meta[df_meta['FinalImageType'] == '2D']

    # -------------------------------------------------------
    # 2. 标签层级逻辑 (Clinical)
    # -------------------------------------------------------
    def determine_cancer_label(row):
        path = row['path_severity'] # 病理
        asses = row['asses']        # BI-RADS
        
        # 1. 病理确诊 -> 优先级最高
        if path in [0, 1]: return 1       # 0=Invasive, 1=In-situ -> 癌症
        elif path in [2, 3, 4, 5, 6]: return 0 # 良性/高危但非癌 -> 非癌症
        
        # 2. 无病理 -> 仅信任 BI-RADS 明确阴性的
        elif pd.isna(path):
            if asses in ['N', 'B', '1', '2']: return 0
        
        # 3. 其他情况 (无病理且 BI-RADS=0/3/4/5) -> 丢弃
        return -1 

    df_clin['Cancer'] = df_clin.apply(determine_cancer_label, axis=1)

    # 再次剔除临床记录中有植入物的
    if 'implanfind' in df_clin.columns:
        df_clin = df_clin[df_clin['implanfind'] != 1]

    # 移除无效标签
    df_clin = df_clin[df_clin['Cancer'] != -1]

    # -------------------------------------------------------
    # 3. 标签聚合
    # -------------------------------------------------------
    # 逻辑：针对同一个病人(empi)、同一次检查(acc)、同一侧乳房(side)聚合
    # 确保同一个病人在不同时间的检查被视为不同的记录
    clin_agg = df_clin.groupby(['empi_anon', 'acc_anon', 'side'])['Cancer'].max().reset_index()
    
    clin_agg.rename(columns={'side': 'ImageLaterality'}, inplace=True)

    # -------------------------------------------------------
    # 4. 合并数据
    # -------------------------------------------------------
    df_merged = pd.merge(df_meta, clin_agg, 
                         on=['empi_anon', 'acc_anon', 'ImageLaterality'], 
                         how='inner')

    # -------------------------------------------------------
    # 5. 生成输出列表
    # -------------------------------------------------------
    output_df = pd.DataFrame()
    
    # [关键修改] 保留 acc_anon 作为 study_id
    # 这样您就可以区分同一个病人的不同次检查了
    output_df['patient_id'] = df_merged['empi_anon'].astype(str)
    output_df['study_id'] = df_merged['acc_anon'].astype(str) 
    
    # 构造 image_id (文件名)
    output_df['image_id'] = df_merged['anon_dicom_path'].apply(
        lambda x: os.path.basename(str(x)).replace('.dcm', '.png')
    )
    
    # 保留原始 DICOM 路径
    output_df['dicom_path'] = df_merged['anon_dicom_path']
    
    output_df['view_position'] = df_merged.get('ViewPosition', 'UNKNOWN')
    output_df['laterality'] = df_merged['ImageLaterality']
    output_df['Cancer'] = df_merged['Cancer'].astype(int)
    output_df['split'] = 'training'

    print(f"筛选完成！有效样本数: {len(output_df)}")
    print(f"正样本(Cancer=1)数量: {output_df['Cancer'].sum()}")
    print(f"独立检查数(Studies): {output_df['study_id'].nunique()}")
    
    output_df.to_csv(OUTPUT_CSV_FOR_CONVERSION, index=False)
    print(f">>> 中间文件已保存至: {OUTPUT_CSV_FOR_CONVERSION}")

if __name__ == "__main__":
    main()