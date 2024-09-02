import torch
import pickle as pk
import os
import numpy as np
from random import shuffle
import random
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.data import Data,Batch
from ogb.graphproppred import PygGraphPropPredDataset

from data_attack_fewshot.attackdata_specified import AttackDataset_specified
from data_pyg.data_pyg import get_dataset
import os.path as osp

# used in pre_train.py
def load_data4pretrain(dataname='CiteSeer', num_parts=200):
    data = pk.load(open('./Dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    # print(data)
    # quit()
    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    hid_dim = input_dim
    graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='./Dataset/{}/'.format(dataname)))


    return graph_list, input_dim, hid_dim


def load4graph(dataset_name, shot_num= 10, num_parts=None):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)
        
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        # 分类并选择每个类别的图
        class_datasets = {}
        for data in dataset:
            label = data.y.item()
            if label not in class_datasets:
                class_datasets[label] = []
            class_datasets[label].append(data)

        train_data = []
        remaining_data = []
        for label, data_list in class_datasets.items():
            train_data.extend(data_list[:shot_num])
            random.shuffle(train_data)
            remaining_data.extend(data_list[shot_num:])

        # 将剩余的数据 1：9 划分为测试集和验证集
        random.shuffle(remaining_data)
        val_dataset_size = len(remaining_data) // 9
        val_dataset = remaining_data[:val_dataset_size]
        test_dataset = remaining_data[val_dataset_size:]
        
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        return input_dim, out_dim, train_data, test_dataset, val_dataset, graph_list



    if  dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts=200

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        
        dataset = list(ClusterData(data=data, num_parts=num_parts))
        graph_list = dataset
        # 这里的图没有标签

        return input_dim, out_dim, graph_list
        # return input_dim, out_dim, None, None, None, graph_list



def load4node_attack_shot_index(dataname, attack_method, shot_num= 10, run_split = 1):
    assert dataname in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv'], 'Currently, attacks are only supported for the specified datasets.'
    atk_type   = attack_method.split('-')[0]
    atk_ptb    = attack_method.split('-')[1]
    path       = osp.expanduser('/home/songsh/MyPrompt/data_pyg/Attack_data')
    dataset    = get_dataset(path, 'Attack-' + dataname, atk_type, atk_ptb)
    print()
    print(f'Attack method : {atk_type} | Attack ptb : {atk_ptb} | Dataset: {dataname}')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    
    # 根据 shot_num 更新训练掩码
    class_counts = {}  # 统计每个类别的节点数
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    # 构建 mask
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    # 这个是随机的取shot方法，这里可以优化一下，因为不同的shot对结果的影响很大，尤其是数据集被攻击的情况下
    index_path = './data_attack/{}/{}/index/shot_{}/{}'.format(dataname, attack_method, str(shot_num), str(run_split))

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

        print(attack_method)
        print(train_indices)
        print(train_lbls)
        # quit()
    # 如果不存在文件夹，则创建shot num索引文件夹 并保存train val test的索引
    else:
        os.makedirs(index_path, exist_ok=True)
        # 存放对应shot num的train val test索引
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y
        # 注意！ seed一样的情况下，不管什么run_split都是一样的！要获得不同的run_split要同时改变seed！
        for label in data.y.unique():
                label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

                # if len(label_indices) < 3 * shot_num:
                #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

                label_indices = label_indices[torch.randperm(len(label_indices))]
                train_indices = label_indices[:shot_num]
                train_mask[train_indices] = True

                remaining_indices = label_indices[shot_num:]
                split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
                
                val_indices = remaining_indices[:split_point]
                test_indices = remaining_indices[split_point:]

                val_mask[val_indices] = True
                test_mask[test_indices] = True

                whole_train_idx.extend(train_indices.numpy())
                whole_val_idx.extend(val_indices.numpy())
                whole_test_idx.extend(test_indices.numpy())

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)

        # shuffled_train_indices = torch.randperm(whole_train_idx.size(0))
        # whole_train_idx = whole_train_idx[shuffled_train_indices]
        whole_train_labels = labels[whole_train_idx]

        # shuffled_val_indices = torch.randperm(whole_val_idx.size(0))
        # whole_val_idx = whole_val_idx[shuffled_val_indices]
        whole_val_labels = labels[whole_val_idx]

        # shuffled_test_indices = torch.randperm(whole_test_idx.size(0))
        # whole_test_idx = whole_test_idx[shuffled_test_indices]
        whole_test_labels = labels[whole_test_idx]

        # 保存文件
        torch.save(whole_train_idx, os.path.join(index_path, 'train_idx.pt'))
        torch.save(whole_train_labels, os.path.join(index_path, 'train_labels.pt'))

        torch.save(whole_val_idx, os.path.join(index_path, 'val_idx.pt'))
        torch.save(whole_val_labels, os.path.join(index_path, 'val_labels.pt'))

        torch.save(whole_test_idx, os.path.join(index_path, 'test_idx.pt'))
        torch.save(whole_test_labels, os.path.join(index_path, 'test_labels.pt'))


    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask


    print(f'len train nodes: {sum(data.train_mask)}')
    print(f'len val   nodes: {sum(data.val_mask)}')
    print(f'len test  nodes: {sum(data.test_mask)}')

    return data, dataset






def load4node_attack_specified_shot_index(data_dir_name, dataname, attack_method, shot_num= 10, run_split = 1):
    assert dataname in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv'], 'Currently, attacks are only supported for the specified datasets.'

    atk_type   = attack_method.split('-')[0]
    atk_ptb    = attack_method.split('-')[1]
    index_path = './{}/{}/shot_{}/{}/index'.format(data_dir_name, dataname, str(shot_num), str(run_split))
    # 首先判断在指定的shot和split下是否存在index

    # 如果存在index，就表示能够根据指定的划分加载攻击后的数据
    if os.path.exists(index_path):
        path       = osp.expanduser('/home/songsh/MyPrompt/{}/{}/shot_{}/{}/'.format(data_dir_name, dataname, shot_num, run_split))
        dataset    = AttackDataset_specified(root = path, name = 'Attack-' + dataname,  attackmethod = atk_type, ptb_rate=atk_ptb) # , transform=T.NormalizeFeatures()
        data = dataset[0]
        # 判断一下被攻击数据的划分方式是否和index_path当中存的划分一样，训练集即可
        train_indices        = torch.load(index_path + '/train_idx.pt').type(torch.long)
        attack_train_indices = data.train_mask.nonzero().squeeze()
        # 对两个tensor进行排序
        sorted_train_indices = torch.sort(train_indices).values
        sorted_attack_train_indices = torch.sort(attack_train_indices).values
        # 判断两个排序后的tensor是否相同
        index_equal = torch.equal(sorted_train_indices, sorted_attack_train_indices)
        if not index_equal:
            raise Exception("The distribution of the attack does not match the specified distribution.")
        else:
            # 这里才是完成了的对指定划分进行了攻击数据的加载
            #############################################
            print("Successfully loaded the attack of dataname with few shot {}, split {}".format(str(shot_num), str(run_split)))
            #############################################
            print(f'Attack method : {atk_type} | Attack ptb : {atk_ptb} | Dataset: {dataname}')
            print('======================')
            print(f'Number of graphs: {len(dataset)}')
            print(f'Number of features: {dataset.num_features}')
            print(f'Number of classes: {dataset.num_classes}')
            print()
            print(data)
            print('===========================================================================================================')
            # Gather some statistics about the graph.
            print(f'Number of nodes: {data.num_nodes}')
            print(f'Number of edges: {data.num_edges}')
            print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
            print(f'Has isolated nodes: {data.has_isolated_nodes()}')
            print(f'Has self-loops: {data.has_self_loops()}')
            print(f'Is undirected: {data.is_undirected()}')

        return data, dataset

    else:
        print("Index for the specified shot and run split does not exist. Generating......")
        path_default = osp.expanduser('/home/songsh/MyPrompt/{}/{}/default/'.format(data_dir_name, dataname))
        dataset      = AttackDataset_specified(root = path_default, name = 'Attack-' + dataname,  attackmethod = atk_type, ptb_rate=0.0) # , transform=T.NormalizeFeatures()

        data = dataset[0]  # Get the first graph object.
        # 表示在指定的shot和split下不存在已经生成的索引，所以要根据默认的数据集自己生成并存放的index当中

        # 创建shot num索引文件夹 并保存train val test的索引

        os.makedirs(index_path, exist_ok=True)
        # 存放对应shot num的train val test索引
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y
        # 注意！ seed一样的情况下，不管什么run_split都是一样的！要获得不同的run_split要同时改变seed！
        for label in data.y.unique():
                label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

                # if len(label_indices) < 3 * shot_num:
                #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

                label_indices = label_indices[torch.randperm(len(label_indices))]
                train_indices = label_indices[:shot_num]

                remaining_indices = label_indices[shot_num:]
                split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
                
                val_indices = remaining_indices[:split_point]
                test_indices = remaining_indices[split_point:]


                whole_train_idx.extend(train_indices.numpy())
                whole_val_idx.extend(val_indices.numpy())
                whole_test_idx.extend(test_indices.numpy())

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)

        # shuffled_train_indices = torch.randperm(whole_train_idx.size(0))
        # whole_train_idx = whole_train_idx[shuffled_train_indices]
        whole_train_labels = labels[whole_train_idx]

        # shuffled_val_indices = torch.randperm(whole_val_idx.size(0))
        # whole_val_idx = whole_val_idx[shuffled_val_indices]
        whole_val_labels = labels[whole_val_idx]

        # shuffled_test_indices = torch.randperm(whole_test_idx.size(0))
        # whole_test_idx = whole_test_idx[shuffled_test_indices]
        whole_test_labels = labels[whole_test_idx]

        # 保存文件
        torch.save(whole_train_idx, os.path.join(index_path, 'train_idx.pt'))
        torch.save(whole_train_labels, os.path.join(index_path, 'train_labels.pt'))

        torch.save(whole_val_idx, os.path.join(index_path, 'val_idx.pt'))
        torch.save(whole_val_labels, os.path.join(index_path, 'val_labels.pt'))

        torch.save(whole_test_idx, os.path.join(index_path, 'test_idx.pt'))
        torch.save(whole_test_labels, os.path.join(index_path, 'test_labels.pt'))

        print(f'len train nodes: {len(whole_train_idx)}')
        print(f'len val   nodes: {len(whole_val_idx)}')
        print(f'len test  nodes: {len(whole_test_idx)}')

        print('train indices: {}'.format(whole_train_idx))

        raise Exception("Generated the specified data split, but it still needs to be attacked.")




def load4node_shot_index(dataname, preprocess_method, shot_num= 10, run_split = 1):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname)#, transform=NormalizeFeatures()) 服了，找了一晚上问题发现在这里 不要加，要和pretrain统一，tmd 你大爷

        # use the largest connected component
        # print('Now use LLC datasets for pretrain !')
        # path    = osp.expanduser('/home/songsh/MyPrompt/data_pyg/Attack_data')
        # dataset = get_dataset(path, 'Attack-' + dataname, 'Meta_Self', 0.0) # 0.0的扰动率即代表最大联通分量 
        # 注意，get_dataset里对特征进行normolize了，所以预训练有问题，已经取消

    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    if preprocess_method == 'svd':
        print('use svd')
        data = pk.load(open('./data/{}_feature_reduced.data'.format(dataname), 'br'))
    else:
        data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

     # 根据 shot_num 更新训练掩码
    class_counts = {}  # 统计每个类别的节点数
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    # 构建 mask
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    # index_path = './data/{}/index/shot_{}/{}'.format(dataname, str(shot_num), str(run_split))
    index_path = './data_fewshot/{}/shot_{}/{}/index'.format(dataname, str(shot_num), str(run_split))
    
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

        # print(train_indices)
        # print(train_lbls)
        # quit()
    # 如果不存在文件夹，则创建shot num索引文件夹 并保存train val test的索引
    else:
        os.makedirs(index_path, exist_ok=True)
        # 存放对应shot num的train val test索引
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y
        # 注意！ seed一样的情况下，不管什么run_split都是一样的！要获得不同的run_split要同时改变seed！
        for label in data.y.unique():
                label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

                # if len(label_indices) < 3 * shot_num:
                #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

                label_indices = label_indices[torch.randperm(len(label_indices))]
                train_indices = label_indices[:shot_num]
                train_mask[train_indices] = True

                remaining_indices = label_indices[shot_num:]
                split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
                
                val_indices = remaining_indices[:split_point]
                test_indices = remaining_indices[split_point:]

                val_mask[val_indices] = True
                test_mask[test_indices] = True

                whole_train_idx.extend(train_indices.numpy())
                whole_val_idx.extend(val_indices.numpy())
                whole_test_idx.extend(test_indices.numpy())

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)

        # shuffled_train_indices = torch.randperm(whole_train_idx.size(0))
        # whole_train_idx = whole_train_idx[shuffled_train_indices]
        whole_train_labels = labels[whole_train_idx]

        # shuffled_val_indices = torch.randperm(whole_val_idx.size(0))
        # whole_val_idx = whole_val_idx[shuffled_val_indices]
        whole_val_labels = labels[whole_val_idx]

        # shuffled_test_indices = torch.randperm(whole_test_idx.size(0))
        # whole_test_idx = whole_test_idx[shuffled_test_indices]
        whole_test_labels = labels[whole_test_idx]

        # 保存文件
        torch.save(whole_train_idx, os.path.join(index_path, 'train_idx.pt'))
        torch.save(whole_train_labels, os.path.join(index_path, 'train_labels.pt'))

        torch.save(whole_val_idx, os.path.join(index_path, 'val_idx.pt'))
        torch.save(whole_val_labels, os.path.join(index_path, 'val_labels.pt'))

        torch.save(whole_test_idx, os.path.join(index_path, 'test_idx.pt'))
        torch.save(whole_test_labels, os.path.join(index_path, 'test_labels.pt'))


    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask


    print(f'len train nodes: {sum(data.train_mask)}')
    print(f'len val   nodes: {sum(data.val_mask)}')
    print(f'len test  nodes: {sum(data.test_mask)}')

    return data, dataset




def load4node_demo1(dataname, preprocess_method, shot_num= 10):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname)#, transform=NormalizeFeatures()) 服了，找了一晚上问题发现在这里 不要加，要和pretrain统一，tmd 你大爷
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    if preprocess_method == 'svd':
        print('use svd')
        data = pk.load(open('./data/{}_feature_reduced.data'.format(dataname), 'br'))
    else:
        data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

     # 根据 shot_num 更新训练掩码
    class_counts = {}  # 统计每个类别的节点数
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    
    for label in data.y.unique():
        label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

        # if len(label_indices) < 3 * shot_num:
        #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")


        label_indices = label_indices[torch.randperm(len(label_indices))]

        train_indices = label_indices[:shot_num]
        train_mask[train_indices] = True

        
        remaining_indices = label_indices[shot_num:]
        split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
        
        val_indices = remaining_indices[:split_point]
        test_indices = remaining_indices[split_point:]

        val_mask[val_indices] = True
        test_mask[test_indices] = True


    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    return data, dataset




def load4link_prediction_single_graph(dataname, num_per_samples=1):
    data, input_dim, output_dim = load4node_demo2(dataname)
    
    # from torch_geometric.utils import remove_self_loops
    # data.edge_index, _ = remove_self_loops(data.edge_index)
    # print(data)
    # quit()
    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
    
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim



def node_degree_as_features(data_list):
    from torch_geometric.utils import degree
    for data in data_list:
        # 计算所有节点的度数，这将返回一个张量
        deg = degree(data.edge_index[0], dtype=torch.long)

        # 将度数张量变形为[nodes, 1]以便与其他特征拼接
        deg = deg.view(-1, 1).float()
        
        # 如果原始数据没有节点特征，可以直接使用度数作为特征
        if data.x is None:
            data.x = deg
        else:
            # 将度数特征拼接到现有的节点特征上
            data.x = torch.cat([data.x, deg], dim=1)


def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)

    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)
        for g in dataset:
            g.y = g.y.squeeze(1)

    data = Batch.from_data_list(dataset)
    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
        
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim



def load4node_demo2(dataname):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        # use raw datasets
        # print('Now use raw datasets for pretrain !')
        # dataset = Planetoid(root='data/Planetoid', name=dataname)#, transform=NormalizeFeatures()) 
        
        # use the largest connected component
        print('Now use LLC datasets for pretrain !')
        path    = osp.expanduser('/home/songsh/MyPrompt/data_pyg/Attack_data')
        dataset = get_dataset(path, 'Attack-' + dataname, 'Meta_Self', 0.0) # 0.0的扰动率即代表最大联通分量 
        # 注意，get_dataset里对特征进行normolize了，所以预训练有问题，已经取消

        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Wisconsin']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Texas']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
        data = dataset[0]
        out_dim = 4
        input_dim = dataset.num_features
        mask = data.y != 1

        # 过滤节点
        data.x = data.x[mask]
        data.y = data.y[mask]

        # 生成新旧索引的映射
        # torch.where(mask) 会返回mask中为True的元素的索引
        old_to_new_indices = torch.zeros(len(mask), dtype=torch.long)  # 初始化映射数组
        old_to_new_indices[mask] = torch.arange(mask.sum())  # 被保留节点的新索引

        # 过滤边
        edge_index = data.edge_index
        mask_edge = mask[edge_index[0]] & mask[edge_index[1]]
        filtered_edge_index = edge_index[:, mask_edge]

        # 更新边索引
        data.edge_index = torch.stack([old_to_new_indices[filtered_edge_index[0]],
                                    old_to_new_indices[filtered_edge_index[1]]], dim=0)
        for index in range(data.y.size(0)):
            if data.y[index] == 4:
                data.y[index] = 1
    elif dataname == 'Actor':
        dataset = Actor(root='data/Actor')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset')
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes

    return data, input_dim, out_dim



# used in pre_train.py
def NodePretrain(dataname='CiteSeer', num_parts=200, preprocess_method = 'svd', split_method='Cluster'):

    data, input_dim, _ = load4node_demo2(dataname)

    # if dataname in ['PubMed', 'CiteSeer', 'Cora']:
    #     dataset = Planetoid(root='data/Planetoid', name=dataname)
    # elif dataname in ['Computers', 'Photo']:
    #     dataset = Amazon(root='data/amazon', name=dataname)
    # elif dataname == 'Reddit':
    #     dataset = Reddit(root='data/Reddit')
    # elif dataname == 'WikiCS':
    #     dataset = WikiCS(root='data/WikiCS')
    # elif dataname == 'Flickr':
    #     dataset = Flickr(root='data/Flickr')
    # data = dataset[0]


    ####### feature svd
    if preprocess_method == 'svd':
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
        pk.dump(data, open('./data/{}_feature_reduced.data'.format(dataname), 'bw'))
        data = pk.load(open('./data/{}_feature_reduced.data'.format(dataname), 'br'))
    ####### feature svd

    if(split_method=='Cluster'):
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif(split_method=='Random Walk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} random walk subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

    else:
        print('None split method!')
        exit()
    
    return graph_list, input_dim

