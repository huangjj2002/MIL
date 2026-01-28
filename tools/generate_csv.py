import pandas as pd
import numpy as np
import os


CLINICAL_FILE = '../EMBED_OpenData_clinical.csv'
METADATA_FILE = '../EMBED_OpenData_metadata.csv'
OUTPUT_CSV_FOR_CONVERSION = 'list_for_conversion.csv' 


def main():
    print(">>> Step 1: 读取数据...")
    df_clin = pd.read_csv(CLINICAL_FILE, low_memory=False)
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)

    print(f"原始记录 -> 临床: {len(df_clin)} | 元数据: {len(df_meta)}")


    # 去除植入物,如果保留植入物的话将!=改为==
    if 'BreastImplantPresent' in df_meta.columns:
        df_meta = df_meta[df_meta['BreastImplantPresent'] != 'YES']
    

    if 'FinalImageType' in df_meta.columns:

        df_meta = df_meta[df_meta['FinalImageType'] == '2D']


    def determine_cancer_label(row):
        path = row['path_severity'] 
        asses = row['asses']      
        
      
        if path in [0, 1]: return 1      
        elif path in [2, 3, 4, 5, 6]: return 0 
        

        elif pd.isna(path):
            if asses in ['N', 'B', '1', '2']: return 0
        
     
        return -1 

    df_clin['Cancer'] = df_clin.apply(determine_cancer_label, axis=1)

    # 剔除临床记录中有植入物的，保留植入物将!=换成==
    if 'implanfind' in df_clin.columns:
        df_clin = df_clin[df_clin['implanfind'] != 1]


    df_clin = df_clin[df_clin['Cancer'] != -1]


    clin_agg = df_clin.groupby(['empi_anon', 'acc_anon', 'side'])['Cancer'].max().reset_index()
    
    clin_agg.rename(columns={'side': 'ImageLaterality'}, inplace=True)

    df_merged = pd.merge(df_meta, clin_agg, 
                         on=['empi_anon', 'acc_anon', 'ImageLaterality'], 
                         how='inner')


    output_df = pd.DataFrame()
    

    output_df['patient_id'] = df_merged['empi_anon'].astype(str)
    output_df['study_id'] = df_merged['acc_anon'].astype(str) 
    
    
    output_df['image_id'] = df_merged['anon_dicom_path'].apply(
        lambda x: os.path.basename(str(x)).replace('.dcm', '.png')
    )
    

    output_df['dicom_path'] = df_merged['anon_dicom_path']
    
    output_df['view_position'] = df_merged.get('ViewPosition', 'UNKNOWN')
    output_df['laterality'] = df_merged['ImageLaterality']
    output_df['Cancer'] = df_merged['Cancer'].astype(int)
    output_df['split'] = 'training'

    print(f"筛选完成！样本数: {len(output_df)}")
    print(f"正样本数量: {output_df['Cancer'].sum()}")
    print(f"独立检查数: {output_df['study_id'].nunique()}")
    
    output_df.to_csv(OUTPUT_CSV_FOR_CONVERSION, index=False)
    print(f">>> 输出文件已保存至: {OUTPUT_CSV_FOR_CONVERSION}")

if __name__ == "__main__":
    main()