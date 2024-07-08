import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
import scipy.sparse as sp
import numpy as np
import scipy
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops

class AttackDataset(InMemoryDataset):
    def __init__(self, root, name, attackmethod, ptb_rate, transform=None, pre_transform=None):  
        # 注意这些要写在super前面，因为raw那些方法都是父类的，super init在调用父类的init的过程中会用搞到raw_dir,raw_file_name这些，如果写后面这些属性还没有定义，就会报错
        self.name         = name.split('-',1)[1].lower() # Cora,Citeseer,Pubmed,Polblogs
        self.attackmethod = attackmethod
        self.ptb_rate     = ptb_rate

        # 数据的下载和处理过程在父类中调用实现
        super(AttackDataset, self).__init__(root, transform, pre_transform)

        # 加载数据
        # self.data, self.slices = torch.load(self.processed_paths[0],map_location= 'cuda') #注意这里需要map_location
        self.data, self.slices = torch.load(self.processed_paths[0], map_location='cpu') # 根据模型需要看要加载到cpu还是gpu上


    @property
    def raw_dir(self) -> str:
        path = osp.join(self.root,self.attackmethod)
        return osp.join(path, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        path = osp.join(self.root,self.attackmethod)
        return osp.join(path, self.name, '{}_processed'.format(self.ptb_rate))


    # 将函数修饰为类属性
    @property
    def raw_file_names(self):
        self.attack_file = '{}_{}_{}'.format(self.attackmethod, self.name, self.ptb_rate)
        return ['{}_features'.format(self.name), '{}_lablels'.format(self.name),
                '{}_idx_train.npy'.format(self.attack_file),
                '{}_idx_val.npy'.format(self.attack_file),
                '{}_idx_test.npy'.format(self.attack_file),
                '{}.pt'.format(self.attack_file)]

    @property
    def processed_file_names(self):
        return ['data_{}_{}_{}.pt'.format(self.name,self.attackmethod,self.ptb_rate)]

    def download(self):
        # download to self.raw_dir
        pass

    def process(self):
        features = self.to_tensor_features(scipy.sparse.load_npz(osp.join(self.raw_dir, '{}_features.npz'.format(self.name)))).to_dense()
        self.labels   = torch.tensor(np.load(osp.join(self.raw_dir, '{}_labels.npy'.format(self.name))))
        adj = torch.load(osp.join(self.raw_dir, '{}.pt'.format(self.attack_file)),map_location= 'cuda') #注意这里需要map_location

        # 有的文件有自环，有的文件没有，所以这里统一的处理一下
        edge_index, _ = remove_self_loops(adj.coalesce().indices())
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))

        # edge_index, _ = add_self_loops(adj.coalesce().indices(), num_nodes=features.size(0))
        
        data = Data(x=features, edge_index=edge_index, y=self.labels)

        self.idx_train = np.load(osp.join(self.raw_dir, '{}_idx_train.npy'.format(self.attack_file)))
        self.idx_val = np.load(osp.join(self.raw_dir, '{}_idx_val.npy'.format(self.attack_file)))
        self.idx_test = np.load(osp.join(self.raw_dir, '{}_idx_test.npy'.format(self.attack_file)))

        self.get_mask()

        data.train_mask = torch.tensor(self.train_mask)
        data.val_mask = torch.tensor(self.val_mask)
        data.test_mask = torch.tensor(self.test_mask)
        
        data = data if self.pre_transform is None else self.pre_transform(data)
        # 这里的save方式以及路径需要对应构造函数中的load操作
        torch.save(self.collate([data]), self.processed_paths[0])


    def to_tensor_features(self,features):
        if sp.issparse(features):
            features = self.sparse_mx_to_torch_sparse_tensor(features)
        else:
            features = torch.FloatTensor(np.array(features))
        return features
    
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow, sparsecol),1)
        sparsedata=torch.FloatTensor(sparse_mx.data)
        return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))
    
    def get_mask(self):
        def get_mask(idx):
            mask = np.zeros(self.labels.shape[0], dtype=np.bool_)
            mask[idx] = 1
            return mask

        self.train_mask = get_mask(self.idx_train)
        self.val_mask = get_mask(self.idx_val)
        self.test_mask = get_mask(self.idx_test)