import numpy as np
from prompt_graph.tasker import NodeTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args


args = get_args()

# 两种统计方式，按照split划分和按照seed划分
# 按照split划分
# 记录每个split在不同seed的值 初始化一个dict
all_split_acc_list = {}
all_split_acc_dict = {}
for split_num in args.run_split:
    all_split_acc_dict[split_num] = {}
    all_split_acc_list[split_num] = []

# 按照seed划分
all_seed_acc_list  = {}
all_seed_acc_dict  = {}

for seed in args.seed:
    seed_everything(seed)
    # 对多次划分进行测试
    if args.task == 'NodeTask':
        # 记录seed对应不同的split
        seed_acc_list = []
        seed_acc_dict = {}
        for split_num in args.run_split:
            tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, hid_dim=args.hid_dim,
                            dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, 
                            prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, run_split= split_num, preprocess_method = args.preprocess_method, attack_downstream = args.attack_downstream, attack_method = args.attack_method, specified = args.specified)
            
            test_acc = tasker.run()

            # 记录每个split在不同seed的值
            all_split_acc_list[split_num].append(test_acc)
            all_split_acc_dict[split_num][seed] = test_acc

            # 记录同一seed下对应不同的split
            seed_acc_list.append(test_acc)
            seed_acc_dict[split_num] = test_acc

        all_seed_acc_list[seed] = seed_acc_list
        all_seed_acc_dict[seed] = seed_acc_dict


print(all_seed_acc_dict)
print(all_split_acc_dict)
print('########################################################################################')
# 打印一个seed下多个split的平均
for seed, seed_acc_dict in all_seed_acc_dict.items():
    for split_num, acc in seed_acc_dict.items():
        print('seed: {} | split {} : {}'.format(seed, split_num, acc))

    seed_final_acc, seed_final_acc_std = np.mean(all_seed_acc_list[seed]), np.std(all_seed_acc_list[seed])
    print(f"# Seed {seed} Muti Split Final Acc: {seed_final_acc:.4f}±{seed_final_acc_std:.4f}")
print('########################################################################################')
# 打印一个split下多个seed的平均
for split_num, split_acc_dict in all_split_acc_dict.items():
    if len(all_split_acc_list[split_num]) ==1:
        print("There's only one result, it's recommended to try several seeds.")
    else:
        # 对所有seed的结果排序，去掉最低和最高的值再求平均
        all_split_acc_list[split_num].sort(reverse=True)
        all_split_acc_list[split_num] = all_split_acc_list[split_num][:-1]
    for seed, acc in split_acc_dict.items():
        print('split: {} | seed {} : {}'.format(split_num, seed, acc))

    split_final_acc, split_final_acc_std = np.mean(all_split_acc_list[split_num]), np.std(all_split_acc_list[split_num])
    print(f"# Split {split_num} Muti Seed Acc without min value: {split_final_acc:.4f}±{split_final_acc_std:.4f}")
print('########################################################################################')


