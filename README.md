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
在根项目下创建models文件夹用于存放模型文件，将用于特征提取的EfficientNet-B2 image encoder存放在models下[EfficientNet-B2下载地址](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar)。将[模型权重下载并存入](https://drive.google.com/file/d/1pcr5wa8cI7R8L-7MfkXBEBB2IE02NmMI/view)一并放入models中。    
对模型进行评估时将交叉验证得到的fold0,fold1,...,fold4复制到models文件夹下。
使用tools下的prep_embed_for_mil_mt_resize.py将dicom文件转换为png。



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
     --data_dir "/mnt/f/data" \
     --img_dir "images_png" \
     --csv_file "embed_data.csv" \
     --dataset "ViNDr" \
     --label "cancer" \
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
     --epochs 1 \
     --lr 1e-5 \
     --batch-size 8 \
     --weighted-BCE "y" \
     --output_dir "one_epoch" \
     --gpu_id 0


   #评估命令
   python main.py \
   --evaluation \
   --eval_set "test" \
   --resume "./models" \
   --clip_chk_pt_path "models/b2-model-best-epoch-10.tar" \
   --data_dir "/mnt/f/data" \
   --img_dir "images_png" \
   --csv_file "test_data.csv" \
   --dataset "ViNDr" \
   --label "cancer" \
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
   --output_dir "result_eval"\
   --n_runs 5
   --gpu_id 0

   ```
   data_dir,img_dir,csv_file分别指向EMBED数据根目录，图片根目录以及csv文件的路径。EMBED数据根目录下应该包含图片根目录以及csv文件路径。output_dir指向模型的输出路径，将训练后得到的fold_0,fold_1,...,fold_4放入models中。
    
   
   

