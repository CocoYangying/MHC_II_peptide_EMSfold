
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

cd MHC_II_peptide_EMSfold/extract_feature_for_train

python extract_inter_features.py \
  --input ../demo_data/rawdata.Feature_extraction_ESMfold/fold1/ \
  --output ../demo_data/data.inter_model/fold1 \
  2>&1 | tee realtime.inter_data_extraction_c000.log

python extract_full_features.py \
--root ../demo_data/rawdata.Feature_extraction_ESMfold/ \
--output_root ../demo_data/data.full_model \
--feature_keys s_z pae contact \
2>&1 | tee realtime.full_data_extraction_train.log

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