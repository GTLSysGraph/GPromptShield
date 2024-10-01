#################################################################################
# 'MultiGprompt' 'GPF'  'GPF-plus' 'RobustPrompt-GPF', 'RobustPrompt-GPFplus', 'RobustPrompt-T', 'GPF-Tranductive', 'GPF-plus-Tranductive'  'All-in-one' 'GPPT' 'Gprompt'
dataset_names=('Citeseer')
prompt_names=('GPPT' )
pretrin_name='GraphMAE'
pre_train_model_path='./pre_trained_model/Citeseer.GraphMAE.GCN.256hidden_dim.pth'
epoch=20
hid_dim=256
shot_nums=(5 10)
run_splits=(1)
seed=1
#################################################################################

atk_methods=('Meta_Self' 'heuristic' 'DICE' 'random')
mtk_ptbs=(0.0 0.25)
atk_ptbs=(0.5)


# mtk_ptbs=(0.0 0.05 0.1 0.15 0.2 0.25) /$pretrin_name
# atk_ptbs=(0.0 0.1 0.2 0.3 0.4 0.5)
# atk_methods=('Meta_Self' 'heuristic' 'DICE' 'random')


process_num=0
max_process_num=3


for dataset_name in "${dataset_names[@]}"; do
    for prompt_name in "${prompt_names[@]}"; do
        for shot_num in "${shot_nums[@]}"; do
            for run_split in "${run_splits[@]}"; do
                for atk_method in "${atk_methods[@]}"; do
                    if [ $atk_method == 'Meta_Self' ]
                    then
                        for atk_ptb in "${mtk_ptbs[@]}"; do
                            echo "运行顺序: $dataset_name $prompt_name $shot_num $run_split $atk_method $atk_ptb"
                    
                            dir="./logs/${prompt_name}/$pretrin_name"
                            if [ ! -d "$dir" ];then
                            mkdir $dir
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
                            --prompt_type $prompt_name \
                            --shot_num $shot_num \
                            --run_split $run_split \
                            --hid_dim $hid_dim \
                            --num_layer 2 \
                            --epochs $epoch \
                            --seed $seed \
                            --attack_downstream \
                            --attack_method "${atk_method}-${atk_ptb}" \
                            > "./logs/${prompt_name}/$pretrin_name/${dataset_name}_shot_${shot_num}_split_${run_split}_${atk_method}_${atk_ptb}" &

                            process_num=`expr $process_num + 1`
                            process_num=`expr $process_num % $max_process_num`
                            if [ $process_num == 0 ]
                            then
                                wait
                            fi 
                        done
                    else
                        for atk_ptb in "${atk_ptbs[@]}"; do
                            echo "运行顺序: $dataset_name $prompt_name $shot_num $run_split $atk_method $atk_ptb"
                    
                            dir="./logs/${prompt_name}/$pretrin_name"
                            if [ ! -d "$dir" ];then
                            mkdir $dir
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
                            --prompt_type $prompt_name \
                            --shot_num $shot_num \
                            --run_split $run_split \
                            --hid_dim $hid_dim \
                            --num_layer 2 \
                            --epochs $epoch \
                            --seed $seed \
                            --attack_downstream \
                            --attack_method "${atk_method}-${atk_ptb}" \
                            > "./logs/${prompt_name}/$pretrin_name/${dataset_name}_shot_${shot_num}_split_${run_split}_${atk_method}_${atk_ptb}" &

                            process_num=`expr $process_num + 1`
                            process_num=`expr $process_num % $max_process_num`
                            if [ $process_num == 0 ]
                            then
                                wait
                            fi 
                        done
                    fi
                done
            done
        done
    done
done