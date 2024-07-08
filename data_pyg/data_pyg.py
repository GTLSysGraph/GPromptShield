import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon,Flickr,Reddit,Yelp, PPI
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset # 会卡住 pip uninstall setuptools 解决
from data_pyg.Attack_data.attackdata import AttackDataset
# from data_pyg.Reddit_small.Reddit_small import Reddit_small

def get_dataset(path, name, attackmethod= None, attackptb = None):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy', 'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Flickr','Yelp','Reddit','Attack-Cora','Attack-Cora_ml','Attack-Citeseer','Attack-Pubmed','Attack-polblogs','PPI','Reddit_small']
    
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('/home/songsh/MyPrompt/data_pyg')


    if name == 'Coauthor-Phy':
        return Coauthor(osp.join(root_path, 'Coauthor'), name='physics', transform=T.NormalizeFeatures())

    if name == 'Coauthor-CS':
        return Coauthor(osp.join(root_path, 'Coauthor'), name='cs', transform=T.NormalizeFeatures())
      
    if name == 'Yelp':
        return Yelp(root=path, transform=T.NormalizeFeatures())

    if name == 'Flickr':
        return Flickr(root=path, transform=T.NormalizeFeatures())

    if name == 'Reddit':
        return Reddit(root=path, transform=T.NormalizeFeatures())

    # if name == 'Reddit_small':
    #     return Reddit_small(root=path, transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    
    if name == 'PPI':
        return [PPI(root=path, split='train'),PPI(root=path, split='val'),PPI(root=path, split='test')]

    if name.startswith('Attack'):
        return AttackDataset(root = path, name = name, attackmethod = attackmethod, ptb_rate=attackptb) # , transform=T.NormalizeFeatures()
        # 这里对特征进行normolize会导致预训练有问题，取消

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name) #S2GAE里不需要, transform=T.NormalizeFeatures()

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

