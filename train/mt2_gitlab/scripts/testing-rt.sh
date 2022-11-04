dataset=round_trip
model=MIT_mixed_augm_model_average_20

python ../translate.py \
-model ../available_models/MIT_mixed_augm/${model}.pt \
-src ../data/${dataset}/src-rt-400k-noreagents.txt \
-output ../results/round_trip/predictions_rt_400k-noreagents.txt \
-batch_size 64 \
-max_length 200 \
-replace_unk \
-gpu 1
