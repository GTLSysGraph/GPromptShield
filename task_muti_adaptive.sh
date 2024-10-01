#################################################################################
# 'MultiGprompt' 'All-in-one' 'GPF'  'GPF-plus' 'RobustPrompt-GPF', 'RobustPrompt-GPFplus', 'RobustPrompt-T', 'GPF-Tranductive', 'GPF-plus-Tranductive'  'GPPT' 'Gprompt'
dataset_names=('Cora_ml')
prompt_names=('Gprompt')
# pretrin_name='GraphMAE'
pre_train_model_path='./pre_trained_model_adaptive/Cora_ml.Edgepred_Gprompt.GCN.64hidden_dim.pth'
epoch=100
hid_dim=64
shot_nums=(5 10)
run_splits=(1)
seed=1
#################################################################################

# atk_methods=('gnn_guard')
# atk_ptbs=(0.1499)

# cora ml
atk_methods=('gnn_guard' 'svd_gcn' 'grand' 'jaccard_gcn')
atk_ptbs=(0.1499 0.1274 0.1499 0.1487)

# citeseer
# atk_methods=('gnn_guard' 'svd_gcn' 'grand' 'jaccard_gcn')
# atk_ptbs=(0.1499 0.12 0.1262 0.1499)

# /$pretrin_name

process_num=0
max_process_num=2

length=${#atk_methods[@]}

for dataset_name in "${dataset_names[@]}"; do
    for prompt_name in "${prompt_names[@]}"; do
        for shot_num in "${shot_nums[@]}"; do
            for run_split in "${run_splits[@]}"; do
                for ((i=0; i<length; i++)); do
                    method=${atk_methods[i]}
                    atk_ptb=${atk_ptbs[i]}
                    echo "运行顺序: $dataset_name $prompt_name $shot_num $run_split $method $atk_ptb"
            
                    dir="./logs_adaptive/${prompt_name}"
                    if [ ! -d "$dir" ];then
                        mkdir -p $dir
                        echo "创建文件夹成功"
                    else
                        echo "文件夹已经存在"
                    fi


                    CUDA_VISIBLE_DEVICES=1
                    python MyTask.py \
                    --pre_train_model_path $pre_train_model_path \
                    --task NodeTask \
                    --dataset_name $dataset_name \
                    --preprocess_method 'none' \
                    --gnn_type 'GCN' \
                    --prompt_type $prompt_name  \
                    --shot_num $shot_num \
                    --run_split $run_split \
                    --hid_dim $hid_dim \
                    --num_layer 2 \
                    --epochs $epoch \
                    --seed $seed \
                    --attack_downstream \
                    --adaptive \
                    --adaptive_scenario 'poisoning' \
                    --adaptive_split 0 \
                    --adaptive_attack_model $method \
                    --adaptive_ptb_rate $atk_ptb \
                    > "./logs_adaptive/${prompt_name}/${dataset_name}_shot_${shot_num}_split_${run_split}_${method}_${atk_ptb}" &

                    process_num=`expr $process_num + 1`
                    process_num=`expr $process_num % $max_process_num`
                    if [ $process_num == 0 ]
                    then
                        wait
                    fi
                done
            done
        done
    done
done