CUDA_VISIBLE_DEVICES=1
python generate_few_shot_attack.py \
--dataset 'Cora' \
--model 'Meta_Self' \
--ptb_rate 0.1 \
--shot_num 1 \
--run_split 1 \