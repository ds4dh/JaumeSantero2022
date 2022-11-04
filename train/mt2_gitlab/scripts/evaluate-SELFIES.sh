dataset=MIT_reactants_pred_w2v_SELFIES
model=${dataset}_model_average_20
python ../score_predictions_SELFIES.py \
-beam_size 1 \
-targets ../data/${dataset}/tgt-test.txt \
-predictions ../results/predictions_${model}_on_${dataset}_test.txt

