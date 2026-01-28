# MIL模型

## 安装 

### 使用虚拟环境安装环境和依赖

1. **安装环境**
   ```bash
   git clone https://github.com/huangjj2002/MIL.git
   cd MIL
   conda env create --name MIL -f environment.yml
   conda activate MIL
   ```
2. **数据处理**  
使用tools文件下的generate_csv.py来对EMBED数据进行处理，要修改CLINICAL_FILE，METADATA_FILE，以及OUTPUT_CSV_FOR_CONVERSION文件，
CLINCAL_FILE应指向EMBED_OpenData_clinical.csv的地址，METADATA_FILE用于指向EMBED_OpenData_metadata.csv的地址，最后OUTPUT_CSV_FOR_CONVERSION用于指定输出的csv的路径。在该脚本一次只能筛选implant或者unimplant样例，需要运行两次才能筛选出implant样例以及非implant样例。
在使用generate_csv.py生成csv文件后使用embed_data.py将dicom文件转换成png文件。INPUT_LIST_CSV指向generate_csv.py生成的csv文件。EMBED_ROOT指向EMBED数据集的根目录，OUTPUT_DIR指向
图片的输出地址，FINAL_CSV_NAME用于指定最终用于模型输入的csv文件的生成路径。  
在根项目下创建models文件夹用于存放模型文件，将用于特征提取的EfficientNet-B2 image encoder存放在models下[EfficientNet-B2下载地址](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar)
在models文件夹再创建一个run_0文件夹，将[模型权重下载并存入](https://drive.google.com/file/d/1pcr5wa8cI7R8L-7MfkXBEBB2IE02NmMI/view)


4. **运行命令**
   ```bash
   #训练命令
   python main.py \
   --train \
   --eval_scheme "kfold_cv+test" \
   --n_folds 5 \
   --training_mode "finetune" \
   --warmup_stage_epochs 0 \
   --warmup-epochs 0 \
   --data_dir "/mnt/embed_data" \
   --img_dir "images_png" \
   --csv_file "embed_data.csv" \
   --dataset "ViNDr" \
   --label "Cancer" \
   --clip_chk_pt_path "./models/b2-model-best-epoch-10.tar" \
   --resume "./models/run_0/best_model.pth" \
   --feature_extraction "online" \
   --mil_type "pyramidal_mil" \
   --multi_scale_model "fpn" \
   --fpn_dim 256 \
   --fcl_encoder_dim 256 \
   --fcl_dropout 0.25 \
   --pooling_type "gated-attention" \
   --drop_attention_pool 0.25 \
   --type_scale_aggregator "gated-attention" \
   --deep_supervision \
   --scales 16 32 128 \
   --epochs 30 \
   --lr 1e-5 \
   --batch-size 8 \
   --weighted-BCE "y" \
   --output_dir "one_epoch"

   #评估命令
   python main.py \
   --evaluation \
   --eval_set "test" \
   --resume "./models" \
   --clip_chk_pt_path "models/b2-model-best-epoch-10.tar" \
   --data_dir "/mnt/picture" \
   --img_dir "implant" \
   --csv_file "data.csv" \
   --dataset "ViNDr" \
   --label "Cancer" \
   --feature_extraction "online" \
   --mil_type "pyramidal_mil" \
   --multi_scale_model "fpn" \
   --fpn_dim 256 \
   --fcl_encoder_dim 256 \
   --fcl_dropout 0.25 \
   --pooling_type "gated-attention" \
   --drop_attention_pool 0.25 \
   --type_scale_aggregator "gated-attention" \
   --deep_supervision \
   --scales 16 32 128 \
   --batch-size 1 \
   --output_dir "result_eval"
   ```
   data_dir,img_dir,csv_file要分别指向EMBED数据根目录，图片根目录以及csv文件的路径。EMBED数据根目录下应该包含图片根目录以及csv文件路径。output_dir指向模型的输出路径，在评估时将模型取出放入models/run_0文件夹中。
    
   
   

