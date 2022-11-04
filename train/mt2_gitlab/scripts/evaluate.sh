dataset=MIT_reactants_pred_200k_noreagents
model=${dataset}_model_average_20
python ../score_predictions.py \
-beam_size 1 \
-targets ../data/${dataset}/tgt-test.txt \
-predictions ../results/predictions_${model}_on_${dataset}_test.txt

