import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
CLINICAL_FILE = '../EMBED_OpenData_clinical.csv'
METADATA_FILE = '../EMBED_OpenData_metadata.csv'
OUTPUT_CSV_FOR_CONVERSION = 'no_implant_list_for_conversion.csv'
# ===========================================

def main():
    df_clin = pd.read_csv(CLINICAL_FILE, low_memory=False)
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)

   
    meta_implant_accs = df_meta[df_meta['BreastImplantPresent'] == 'YES']['acc_anon'].unique()
    clin_implant_accs = df_clin[df_clin['implanfind'] == 1]['acc_anon'].unique()
    
 
    implant_blacklist = set(meta_implant_accs) | set(clin_implant_accs)
    


    df_meta_clean = df_meta[~df_meta['acc_anon'].isin(implant_blacklist)].copy()
    
    if 'FinalImageType' in df_meta_clean.columns:
        df_meta_clean = df_meta_clean[df_meta_clean['FinalImageType'] == '2D']

  
    def determine_cancer_label(row):
        path = row['path_severity']
        asses = row['asses']
        
 
        if path in [0, 1]: return 1
        elif path in [2, 3, 4, 5, 6]: return 0
        

        elif pd.isna(path):
            if str(asses).strip().upper() in ['N', 'B', '1', '2']: return 0
            
        return -1 

    df_clin['Cancer'] = df_clin.apply(determine_cancer_label, axis=1)
    
   
    df_clin_clean = df_clin[~df_clin['acc_anon'].isin(implant_blacklist)]
    clin_agg = df_clin_clean[df_clin_clean['Cancer'] != -1].groupby(['empi_anon', 'acc_anon', 'side'])['Cancer'].max().reset_index()
    clin_agg.rename(columns={'side': 'ImageLaterality'}, inplace=True)


    df_merged = pd.merge(df_meta_clean, clin_agg, 
                         on=['empi_anon', 'acc_anon', 'ImageLaterality'], 
                         how='inner')

    if df_merged.empty:
        return

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

    print(f"无植入物的影像数: {len(output_df)}")
    print(f"Cancer=1数量: {output_df['Cancer'].sum()}")
    print(f"anon数: {output_df['study_id'].nunique()}")
    
    output_df.to_csv(OUTPUT_CSV_FOR_CONVERSION, index=False)
    print(f">>> 结果已保存至: {OUTPUT_CSV_FOR_CONVERSION}")

if __name__ == "__main__":
    main()