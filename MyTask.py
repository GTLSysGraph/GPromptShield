import numpy as np
from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4graph, load4link, induced_graphs_from_edges, CustomTUDataset
import torch

args = get_args()


if args.task == 'NodeTask':
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
        # 记录seed对应不同的split
        seed_acc_list = []
        seed_acc_dict = {}
        for split_num in args.run_split:
            tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, hid_dim=args.hid_dim,
                            dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, 
                            prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, run_split= split_num, preprocess_method = args.preprocess_method, attack_downstream = args.attack_downstream, attack_method = args.attack_method, specified = args.specified, adaptive = args.adaptive, adaptive_scenario=args.adaptive_scenario, adaptive_split= args.adaptive_split, adaptive_attack_model= args.adaptive_attack_model, adaptive_ptb_rate= args.adaptive_ptb_rate)
            
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








elif args.task == 'GraphTask':
    for seed in args.seed:
        seed_everything(seed)

        input_dim, output_dim, dataset = load4graph(args.dataset_name)
        tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim, task_type = 'GraphTask')
        _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _= tasker.run()

elif args.task == 'LinkTask': # 链接预测任务转换为图任务，只是induced graph不同
    for seed in args.seed:
        seed_everything(seed)
        assert args.dataset_name in ['Cora', 'Citeseer', 'PubMed']
        dataset = load4link(args.dataset_name)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        dataset = induced_graphs_from_edges(data, args.device, smallest_size=1, largest_size=30)
        dataset = CustomTUDataset(dataset)
        
        tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                    shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = 1024, dataset = dataset, input_dim = input_dim, output_dim = 2, task_type = 'LinkTask')
        
        _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _= tasker.run()