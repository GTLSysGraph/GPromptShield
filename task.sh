
CUDA_VISIBLE_DEVICES=1
python MyTask.py \
--pre_train_model_path './pre_trained_model/Cora.GraphCL.GCN.256hidden_dim.pth' \
--task NodeTask \
--dataset_name 'Cora' \
--preprocess_method 'none' \
--gnn_type 'GCN' \
--prompt_type 'RobustPrompt-I' \
--shot_num 10 \
--run_split 1 \
--hid_dim 256 \
--num_layer 2 \
--epochs 100 \
--seed 1 \
--attack_downstream \
--attack_method 'Meta_Self-0.25' \
# --specified

# 注意！ 因为在Mytask.py中的开始有seed_everything(seed)！所以在seed一样的情况下，不管什么run_split都是一样的！在第一次生成run_split的时候要同时改变seed！ 让run_split和seed一起改变，生成之后就直接读取了，就不用管seed了

# --seed 454