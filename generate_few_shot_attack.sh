CUDA_VISIBLE_DEVICES=1
python generate_few_shot_attack.py \
--dataset 'Citeseer' \
--model 'Meta_Self' \
--ptb_rate 0.0 \
--shot_num 10 \
--run_split 1 \