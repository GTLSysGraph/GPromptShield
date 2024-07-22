from prompt_graph.data import load4node_attack_shot_index
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import scipy
import os
import os.path as osp
from data_attack_fewshot.attackdata_specified import AttackDataset_specified

# os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda',        action='store_true',        default=False,                                                                          help='Disables CUDA training.')
parser.add_argument('--seed',           type=int,                   default=15,                                                                             help='Random seed.')
parser.add_argument('--epochs',         type=int,                   default=200,                                                                            help='Number of epochs to train.')
parser.add_argument('--lr',             type=float,                 default=0.01,                                                                           help='Initial learning rate.')
parser.add_argument('--weight_decay',   type=float,                 default=5e-4,                                                                           help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden',         type=int,                   default=16,                                                                             help='Number of hidden units.')
parser.add_argument('--dropout',        type=float,                 default=0.5,                                                                            help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset',        type=str,                   default='Cora',  choices=['Cora', 'CiteSeer', 'PubMed',],      help='dataset')
parser.add_argument('--ptb_rate',       type=float,                 default= 0.0,           help='pertubation rate')
parser.add_argument('--model',          type=str,                   default='Meta_Self', choices=['Meta_Self', 'A_Meta_Self', 'Meta_Train', 'A_Meta_Train'], help='model variant')
# add by ssh
parser.add_argument('--shot_num',       type=int,                   default=1,                                                                             help='shot num.')
parser.add_argument('--run_split',      type=int,                   default=1,                                                                             help='run_split.')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)



########################################################################################################
########################################################################################################
########################################################################################################
# 加载few shot的clean data 并根据在prompt中生成的索引构建train/val/test的mask
path_default = osp.expanduser('/home/songsh/MyPrompt/data_attack_fewshot/{}/default/'.format(args.dataset))
dataset      = AttackDataset_specified(root = path_default, name = 'Attack-' + args.dataset,  attackmethod = "Meta_Self", ptb_rate=0.0) # , transform=T.NormalizeFeatures()
data = dataset[0]  # Get the first graph object.

index_path = './data_attack_fewshot/{}/shot_{}/{}/index'.format(args.dataset, str(args.shot_num), str(args.run_split))
# 构建 mask
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

# 加载在方法中生成的指定shot和指定split的索引
if os.path.exists(index_path):
    train_indices  = torch.load(index_path + '/train_idx.pt').type(torch.long)
    train_lbls     = torch.load(index_path + '/train_labels.pt').type(torch.long).squeeze()

    val_indices    = torch.load(index_path + '/val_idx.pt').type(torch.long)
    val_lbs        = torch.load(index_path + '/val_labels.pt').type(torch.long).squeeze()

    test_indices   = torch.load(index_path + '/test_idx.pt').type(torch.long)
    test_lbls      = torch.load(index_path + '/test_labels.pt').type(torch.long).squeeze()

    train_mask[train_indices]  =  True
    val_mask[val_indices]      =  True
    test_mask[test_indices]    =  True

    print(train_indices)
    print(train_lbls)

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
else:
    raise Exception("Please first generate the specified | shot {} and run split {} | data split index in the prompt.".format(str(args.shot_num), str(args.run_split)))
########################################################################################################
########################################################################################################
########################################################################################################
# 在加载好了指定的数据划分之后即可进行对应数据划分的攻击


adj  = sp.csr_matrix((np.ones(data.edge_index.shape[1]), (np.array(data.edge_index[0]), np.array(data.edge_index[1]))), shape=[data.x.shape[0], data.x.shape[0]])
features = data.x
labels   = data.y
idx_train  = data.train_mask.nonzero().squeeze()
idx_val    = data.val_mask.nonzero().squeeze()
idx_test   = data.test_mask.nonzero().squeeze()
idx_unlabeled = np.union1d(idx_val, idx_test)


perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)


# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
modified_adj = model.modified_adj


#save
path = '/home/songsh/MyPrompt/data_attack_fewshot/{}/shot_{}/{}/Meta_Self/raw'.format(args.dataset, args.shot_num, args.run_split)
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

torch.save(modified_adj.to_sparse(),   os.path.join(path, '%s_%s_%s.pt' % (args.model, args.dataset,args.ptb_rate)))
if args.ptb_rate == 0: #保存一次label和features即可
    scipy.sparse.save_npz(os.path.join(path, '%s_features' % (args.dataset)), scipy.sparse.csr_matrix(np.array(features)))
    np.save(os.path.join(path, '%s_labels' % (args.dataset)),   labels)

np.save(os.path.join(path, '%s_%s_%s_idx_train' % (args.model, args.dataset,args.ptb_rate)),   idx_train)
np.save(os.path.join(path, '%s_%s_%s_idx_val'   % (args.model, args.dataset,args.ptb_rate)),     idx_val)
np.save(os.path.join(path, '%s_%s_%s_idx_test'  % (args.model, args.dataset,args.ptb_rate)),    idx_test)