# 'MultiGprompt' 'GPF'  'GPF-plus' 'RobustPrompt-GPF', 'RobustPrompt-GPFplus', 'RobustPrompt-T', 'GPF-Tranductive', 'GPF-plus-Tranductive'  'All-in-one' 'GPPT' 'Gprompt'
CUDA_VISIBLE_DEVICES=1
python MyTask.py \
--pre_train_model_path './pre_trained_model/Citeseer.GraphMAE.GCN.256hidden_dim.pth' \
--task NodeTask \
--dataset_name 'Citeseer' \
--preprocess_method 'none' \
--gnn_type 'GCN' \
--prompt_type 'RobustPrompt-T'   \
--shot_num 10 \
--run_split 1 \
--hid_dim 256 \
--num_layer 2 \
--epochs 100 \
--seed 3 \
--attack_downstream \
--attack_method 'DICE-0.5' \
# --adaptive \
# --adaptive_scenario 'poisoning' \
# --adaptive_split 0 \
# --adaptive_attack_model 'gnn_guard' \
# --adaptive_ptb_rate 0.0



# unit test
# --adaptive \
# --adaptive_scenario     'poisoning' \
# --adaptive_split        0 \
# --adaptive_attack_model ='gnn_guard' \
# --adaptive_ptb_rate     0.1499


# common attack
# --attack_method 'DICE-0.5' \



# > "./logs/GPF-plus/Citeseer_shot_5_split_1_Meta_Self_0.0"
# 注意！ 因为在Mytask.py中的开始有seed_everything(seed)！所以在seed一样的情况下，不管什么run_split都是一样的！在第一次生成run_split的时候要同时改变seed！ 让run_split和seed一起改变，生成之后就直接读取了，就不用管seed了


