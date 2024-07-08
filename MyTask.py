import numpy as np
from prompt_graph.tasker import NodeTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args


args = get_args()
seed_everything(args.seed)

# 对多次划分进行测试
if args.task == 'NodeTask':
    acc_list = []
    acc_dict = {}
    for split_num in args.run_split:
        tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, hid_dim=args.hid_dim,
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, 
                        prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, run_split= split_num, preprocess_method = args.preprocess_method, attack_downstream = args.attack_downstream, attack_method = args.attack_method)
        
        test_acc = tasker.run()

        acc_list.append(test_acc)
        acc_dict[split_num] = test_acc
    


    for split_num, acc in acc_dict.items():
        print('split {} : {}'.format(split_num, acc))


    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# Muti Split Final Acc: {final_acc:.4f}±{final_acc_std:.4f}")