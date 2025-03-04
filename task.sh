# 'MultiGprompt' 'GPF'  'GPF-plus' 'RobustPrompt-GPF', 'RobustPrompt-GPFplus', 'RobustPrompt-T', 'RobustPrompt-I','GPF-Tranductive', 'GPF-plus-Tranductive'  'All-in-one' 'GPPT' 'Gprompt'
CUDA_VISIBLE_DEVICES=1
python MyTask.py \
--pre_train_model_path './pre_trained_model_raw/Cora_ml.GraphCL.GCN.64_hidden_dim.pth' \
--task NodeTask \
--dataset_name 'Cora_ml' \
--preprocess_method 'none' \
--gnn_type 'GCN' \
--prompt_type 'RobustPrompt-GPFplus' \
--shot_num 5 \
--run_split 1 \
--hid_dim 64 \
--num_layer 2 \
--epochs 200 \
--seed 62623 \
--attack_downstream \
--attack_method 'heuristic-0.5' \


# unit test
# --adaptive \
# --adaptive_scenario     'poisoning' \
# --adaptive_split        0 \
# --adaptive_attack_model 'gnn_guard' \
# --adaptive_ptb_rate     0.1499


# common attack
# --attack_downstream \
# --attack_method 'Meta_Self-0.0' \



# > "./logs/GPF-plus/Citeseer_shot_5_split_1_Meta_Self_0.0"
# attention! seed change when generate split


