import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--task', type = str)
    parser.add_argument('--dataset_name', type=str, default='Cora',help='Choose the dataset of pretrainor downstream task')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="GCN", help='We support gnn like \GCN\ \GAT\ \GT\ \GCov\ \GIN\ \GraphSAGE\, please read ProG.model module')
    parser.add_argument('--prompt_type', type=str, default="All-in-one", 
                        help='Choose the prompt type for node or graph task, for node task,we support \GPPT\, \All-in-one\, \Gprompt\ for graph task , \All-in-one\, \Gprompt\, \GPF\, \GPF-plus\ ')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='hideen layer of GNN dimensions (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--shot_num', type=int, default = 5, help='Number of shots')
    parser.add_argument('--pre_train_model_path', type=str, default='None', 
                        help='add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of GNN message passing layers (default: 3).')

    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')

    # parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--seed', nargs='+', type=int, help = "Seed for splitting dataset.")
    parser.add_argument('--run_split', nargs='+', type=int, help= "run split num")
    # parser.add_argument('--run_split', type=int, help= "run split num")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 0, help='Number of workers for dataset loading')
    parser.add_argument('--num_layers', type=int, default = 1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default = 5, help='The number of independent basis for GPF-plus')

    # add ssh
    parser.add_argument('--preprocess_method', type=str, default='None', help='Choose preprocess method svd')
    parser.add_argument('--attack_downstream', action='store_true',      default=False, help='Attack Downstream Task')
    parser.add_argument('--specified',         action='store_true',      default=False, help='Attack specified split, Used for some distribution-based attacks')
    # ArgumentParser在传布尔类型变量时，传入参数按字符串处理，所以无论传入什么值，参数值都为True。
    parser.add_argument('--attack_method',     type=  str,               default=  'None') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    # 如果用自适应攻击，就使用下面的参数
    parser.add_argument('--adaptive',                       action='store_true',      default= False, help='Unit Test')
    parser.add_argument('--adaptive_scenario',              type=str,                 default= 'None') # 'poisoning'
    parser.add_argument('--adaptive_split',                 type=int,                 default= 0)
    parser.add_argument('--adaptive_attack_model',          type=str,                 default= 'None') 
    parser.add_argument('--adaptive_ptb_rate',              type=float,               default=0.)




    args = parser.parse_args()
    return args