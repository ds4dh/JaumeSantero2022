dataset=MIT_reactants_pred_300k_noreagents
models=../available_models/${dataset}/checkpoints/*
output=../available_models/${dataset}/${dataset}_model_average_20.pt

python ../tools/average_models.py \
-models ${models} \
-output ${output} 
