dataset=MIT_reactants_pred_300k_noreagents
model=${dataset}_model_average_20

python ../translate.py \
-model ../available_models/${dataset}/${model}.pt \
-src ../data/${dataset}/src-test.txt \
-output ../results/predictions_${model}_on_${dataset}_test.txt \
-batch_size 64 \
-max_length 200 \
-replace_unk \
-gpu 0
