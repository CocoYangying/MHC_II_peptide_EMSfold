# MHC_II_peptide_EMSfold
Leveraged ESM-fold predicted structures (including residue feature matrics s_z, predicted aligned error (PAE),and contact maps) to build an MHC class II binding prediction model integrating multi-source structural information.

Three architectures were tried to get explore which model can effectively learn the patterns of MHC class II binding mode.
1. First model is named inter model, which calculated mean values of each dimensionality.
2. Second model is named full model, using the raw interaction matrics as input without averaging.
3. Third model is named full_bal model, using the raw interaction matrics as input and also balencing the dimensionalities of three feature types(s_z, PAS and contact).
Model full_bal achieved a state-of-the-art accuracy of 67.5% on a held-out test set, outperforming the established NetMHCIIpan tool (58.8% accuracy).

## Step1: extacting structure information by ESMfold:
```
cd MHC_II_peptide_EMSfold/extration_feature_by_esmfold
python esmfold_feature_extraction_multiproc.py \
  --input_csv ../demo_data/rawdata.Feature_extraction_ESMfold/demo.rawinputdata.csv \
  --output_csv ../demo_data/rawdata.Feature_extraction_ESMfold/demo.rawinputdata.esm.csv  \
  --feature_dir ../demo_data/rawdata.Feature_extraction_ESMfold/demo.rawinputdata.esmfold_feature/  \
  --batch_size 1 \
  --recycle 0 \
  --linker_len 25 \
  --chunk_size 32 \
  --group_size 5000 \
  --workers 4 \
  2>&1 | tee run_extract_esmfold.log
```
## Step2.1: getting data for training of inter model
```
cd MHC_II_peptide_EMSfold/extract_feature_for_train
python extract_inter_features.py \
  --input ../demo_data/rawdata.Feature_extraction_ESMfold/fold1/ \
  --output ../demo_data/data.inter_model/fold1 \
  2>&1 | tee realtime.inter_data_extraction_c000.log
```
## Step2.2: getting data for training of full model
```
cd MHC_II_peptide_EMSfold/extract_feature_for_train
python extract_full_features.py \
--root ../demo_data/rawdata.Feature_extraction_ESMfold/ \
--output_root ../demo_data/data.full_model \
--feature_keys s_z pae contact \
2>&1 | tee realtime.full_data_extraction_train.log
```
## Step3.1: Training, inference and evaluation of full_bal model
```
cd MHC_II_peptide_EMSfold/models/full_bal
torchrun --standalone --nnodes=1 --nproc-per-node=4 cv_fullmat_transformer_full_bal.py \
    --root ../../demo_data/data.full_model/ \
    --feature_keys s_z pae contact \
    --num_workers 16 \
    --epochs 100 \
    --batch_size 32 \
    --max_len 15000 \
    --lr 0.00005 \
    --use_flash \
    --patience 20 \
    --lr_step_size 5 \
    --use_amp \
    --dropout 0.3 \
    --save_dir ./checkpoint \
    2>&1 | tee realtime.train_full_bal.log

python infer_eval_full_bal.py \
  --mode fullmat \
  --checkpoint ./checkpoint/full_bal_fold5_best.pt \
  --test_csv ../../demo_data/data.full_model/fold1/fold1_merged.csv \
  --batch_size 64 \
  --threshold 0.324 \
  --out_dir full_bal_output \
  2>&1 | tee realtime.inference_full_bal_model.log
```
## Step3.2: Training, inference and evaluation of inter model
```
cd MHC_II_peptide_EMSfold/models/inter
torchrun --standalone --nnodes=1 --nproc-per-node=4 cv_inter_transformer.py \
    --root ../../demo_data/data.inter_model/ \
    --epochs 100 \
    --batch_size 512 \
    --lr 0.001 \
    --save_dir ./checkpoint/ \
    --seed 42 \
    --patience 20 \
    2>&1 | tee realtime.train_inter.log

python infer_eval.py \
  --mode inter \
  --checkpoint ./checkpoint/inter_fold5_best.pt \
  --test_csv ../../demo_data/data.inter_model/fold1/processed2_c000_inter_demo.esm.csv \
  --batch_size 64 \
  --threshold 0.245 \
  --out_dir infer_output \
  2>&1 | tee realtime.inference_inter_model.log
```

## Step3.3: Training, inference and evaluation of full model
```
cd MHC_II_peptide_EMSfold/models/full
torchrun --standalone --nnodes=1 --nproc-per-node=4 cv_fullmat_transformer_full.py \
    --root ../../demo_data/data.full_model/ \
    --feature_keys s_z pae contact \
    --num_workers 16 \
    --epochs 100 \
    --batch_size 32 \
    --max_len 15000 \
    --lr 0.00005 \
    --use_flash \
    --patience 20 \
    --lr_step_size 5 \
    --use_amp \
    --dropout 0.3 \
    --save_dir ./checkpoint \
    2>&1 | tee realtime.train_full.log


python infer_eval_full.py \
  --mode fullmat \
  --checkpoint ./checkpoint/full_fold5_best.pt \
  --test_csv ../../demo_data/data.full_model/fold1/fold1_merged.csv \
  --batch_size 64 \
  --threshold 0.324 \
  --out_dir full_output \
  2>&1 | tee realtime.inference_full_model.log
```
