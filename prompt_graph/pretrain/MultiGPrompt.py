import torch
import torch.nn as nn
import torch.nn.functional as F
from prompt_graph.data import load4graph
from prompt_graph.prompt import DGI,GraphCL,Lp,AvgReadout, DGIprompt,GraphCLprompt,Lpprompt, GcnLayers

import tqdm
import scipy.sparse as sp
import numpy as np
from prompt_graph.utils import process
import prompt_graph.utils.aug as aug
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tqdm import trange

# 节点分类用这个
class NodePrePrompt(nn.Module):
    def __init__(self, dataset_name, n_h, activation, a1, a2, a3, a4, num_layers_num, p, device):
        super(NodePrePrompt, self).__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.hid_dim = n_h
        n_in, self.nb_nodes = self.load_data()
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, p)
        self.read = AvgReadout()
        self.weighted_feature=weighted_feature(a1,a2,a3)
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.dgiprompt = DGIprompt(n_in, n_h, activation)
        self.graphcledgeprompt = GraphCLprompt(n_in, n_h, activation)
        self.lpprompt = Lpprompt(n_in, n_h)
        sample = self.negetive_sample
        self.sample = torch.tensor(sample,dtype=int).to(self.device)
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()


    def load_data(self):
        self.adj, features, self.labels = process.load_data(self.dataset_name)
        # self.adj, features, self.labels = process.load_data(self.dataset_name)  
        self.features, _ = process.preprocess_features(features)

        if self.dataset_name in ['Texas','Wisconsin']:
            self.negetive_sample = prompt_pretrain_sample(self.adj,50)
        else:
            self.negetive_sample = prompt_pretrain_sample(self.adj,200)
        # Cora为例子的shape (2485, 201)
        # prompt_pretrain_sample为图中的每个节点提供了一个正样本和多个负样本的索引
        nb_nodes = self.features.shape[0]  # node number
        ft_size = self.features.shape[1]  # node features dim
        nb_classes = self.labels.shape[1]  # classes = 6
        return ft_size, nb_nodes

    #  seq1,          seq2,               seq3,                  seq4,                  seq5,              seq6,
    # features,     shuf_fts,        aug_features1edge,    aug_features2edge,    aug_features1mask,  aug_features2mask,
    
    #   adj,      aug_adj1edge,        aug_adj2edge,         aug_adj1mask,          aug_adj2mask,
    # sp_adj,    sp_aug_adj1edge,    sp_aug_adj2edge,      sp_aug_adj1mask,       sp_aug_adj2mask, 
    
    # sparse,        msk,              samp_bias1,           samp_bias2,                lbl
    # sparse,       None,                None,                  None,                 lbl=lbl

    def forward(self, seq1, seq2, seq3, seq4, seq5, seq6, adj, aug_adj1edge, aug_adj2edge, aug_adj1mask, aug_adj2mask,
                sparse, msk, samp_bias1, samp_bias2, lbl):
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        # dual-prompt mechanism    composed prompts和open prompts应用于预训练的图编码器的不同层
        # composed prompts 组合提示旨在将借口任务特定知识转移到下游任务
        logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        # print("logits1", logits1.shape) logits1 torch.Size([1, 4970])
        logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        # print("logits2", logits2.shape) logits2 torch.Size([1, 4970])
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        # print("logits3", logits3.shape) logits3 torch.Size([2485, 256])
        # quit()
        
        # open prompts 开放式提示有助于全局任务间知识的转移 在预训练的时候这两个一起训练，prompt的时候只tune open prompts
        logits4 = self.dgiprompt(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits5 = self.graphcledgeprompt(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits6 = self.lpprompt(self.gcn,seq1,adj,sparse)


        logits11 = logits1 + self.a4*logits4
        logits22 = logits2 + self.a4*logits5
        logits33 = logits3 + self.a4*logits6

        dgiloss = self.loss(logits11, lbl)
        graphcledgeloss = self.loss(logits22, lbl)
        lploss = compareloss(logits33,self.sample,temperature=1.5, device = self.device)
        lploss.requires_grad_(True)
        
        ret = self.a1 * dgiloss + self.a2 * graphcledgeloss + self.a3 * lploss

        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
    
    def pretrain(self):
        batch_size = 1
        nb_epochs = 1000
        patience = 20
        lr = 0.0001
        l2_coef = 0.0
        # hid_units = 256
        sparse = True

        features = torch.FloatTensor(self.features[np.newaxis])
        # 将features数组转换为PyTorch的FloatTensor类型，并增加一个新的维度
        '''
        # ------------------------------------------------------------
        # edge node mask subgraph
        # ------------------------------------------------------------
        '''
        # print("Begin Aug:[{}]".format(args.aug_type))
        # if args.aug_type == 'edge':
        adj = self.adj
        # 边扰动，特征不变
        aug_features1edge = features
        aug_features2edge = features

        drop_percent = 0.1
        aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
        aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges

        # 特征扰动，边不变
        aug_features1mask = aug.aug_random_mask(features, drop_percent=drop_percent)
        aug_features2mask = aug.aug_random_mask(features, drop_percent=drop_percent)

        aug_adj1mask = adj
        aug_adj2mask = adj

        '''
        # ------------------------------------------------------------
        '''

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
        aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

        aug_adj1mask = process.normalize_adj(aug_adj1mask + sp.eye(aug_adj1mask.shape[0]))
        aug_adj2mask = process.normalize_adj(aug_adj2mask + sp.eye(aug_adj2mask.shape[0]))

        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
        sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)

        sp_aug_adj1mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj1mask)
        sp_aug_adj2mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj2mask)

        labels = torch.FloatTensor(self.labels[np.newaxis])
        # print("labels",labels)
        print("adj",sp_adj.shape)
        print("feature",features.shape)
        LP = False
        print("")
        lr=0.0001

        # for name, paramer in self.named_parameters():
        #     print(name, paramer)
        # quit()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available():
            print('Using CUDA')
            self = self.to(self.device)
            features = features.to(self.device)
            aug_features1edge = aug_features1edge.to(self.device)
            aug_features2edge = aug_features2edge.to(self.device)
            aug_features1mask = aug_features1mask.to(self.device)
            aug_features2mask = aug_features2mask.to(self.device)
     
            sp_adj = sp_adj.to(self.device)
            sp_aug_adj1edge = sp_aug_adj1edge.to(self.device)
            sp_aug_adj2edge = sp_aug_adj2edge.to(self.device)
            sp_aug_adj1mask = sp_aug_adj1mask.to(self.device)
            sp_aug_adj2mask = sp_aug_adj2mask.to(self.device)

            labels = labels.to(self.device)
  
    
        cnt_wait = 0
        best = 1e9

        # begin training
        for epoch in range(nb_epochs):
            self.train()
            optimizer.zero_grad()
            idx = np.random.permutation(self.nb_nodes)
            shuf_fts = features[:, idx, :]
            lbl_1 = torch.ones(batch_size, self.nb_nodes)
            lbl_2 = torch.zeros(batch_size, self.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if torch.cuda.is_available():
                shuf_fts = shuf_fts.to(self.device)
                lbl = lbl.to(self.device)
            loss = self(features, shuf_fts, aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
                        sp_adj if sparse else adj,
                        sp_aug_adj1edge if sparse else aug_adj1edge,
                        sp_aug_adj2edge if sparse else aug_adj2edge,
                        sp_aug_adj1mask if sparse else aug_adj1mask,
                        sp_aug_adj2mask if sparse else aug_adj2mask,
                        sparse, None, None, None, lbl=lbl)
            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, nb_epochs, loss.item()))
            loss.backward()
            optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('-' * 100)
                print('Early stopping at '+str(epoch) +' eopch!')
                break

        # folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        torch.save(self.state_dict(),
                    "./pre_trained_model/{}.{}.{}hidden_dim.epoch_{}.pth".format(self.dataset_name, 'MultiGprompt', self.hid_dim, nb_epochs))
        print("+++model saved ! {}.{}.{}hidden_dim.epoch_{}.pth".format(self.dataset_name, 'MultiGprompt', self.hid_dim, nb_epochs))



































# 图分类用这个
class GraphPrePrompt(nn.Module):
    def __init__(self, dataset_name, n_h, activation,a1,a2,a3,num_layers_num,p,device):
        super(GraphPrePrompt, self).__init__()
        print(dataset_name)
        # add here
        n_in, n_out, graph = load4graph(dataset_name)
        self.graph_list = graph
        self.loader = self.get_loader()
        self.dataset_name = dataset_name
        self.device = device
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()
        self.input_dim = n_in
        self.output_dim = n_out
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        self.loss = nn.BCEWithLogitsLoss()

    def get_loader(self):
        loader = DataLoader(self.graph_list, batch_size = 32, shuffle=True,drop_last=True)
        return loader
    
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, 
                sparse, msk, samp_bias1, samp_bias2,
                lbl,sample):
        negative_sample = torch.tensor(sample,dtype=int).to(self.device)
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        dgiloss = self.loss(logits1, lbl)
        graphcledgeloss = self.loss(logits2, lbl)
        lploss = compareloss(logits3,negative_sample,temperature=1.5, device = self.device)
        lploss.requires_grad_(True)
        
        ret =self.a1*dgiloss+self.a2*graphcledgeloss+self.a3*lploss
        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


    def pretrain(self):
        best = 1e9
        self.to(self.device)
        for epoch in range(1000):
            loss = 0
            regloss = 0
            drop_percent=0.1
            patience = 20
            # train_bar = tqdm(self.loader) 
            for step, batch in enumerate(self.loader):

                features,adj =  process.process_tu(batch, self.output_dim, self.input_dim)
                negetive_sample = tu_prompt_pretrain_sample(adj,50)
                nb_nodes = features.shape[0]  # node number
                features = torch.FloatTensor(features[np.newaxis])

                
                aug_features1edge = features
                aug_features2edge = features

                aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
                aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges


                adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
                aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
                aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

                adj = (adj + sp.eye(adj.shape[0])).todense()
                aug_adj1edge = (aug_adj1edge + sp.eye(aug_adj1edge.shape[0])).todense()
                aug_adj2edge = (aug_adj2edge + sp.eye(aug_adj2edge.shape[0])).todense()

        
                adj = torch.FloatTensor(adj[np.newaxis])
                aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
                aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])

                optimiser = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0)
                if torch.cuda.is_available() :
                    # print('Using CUDA')
                    # model = torch.nn.DataParallel(model, device_ids=[0,1]).to(self.device)
                    features = features.to(self.device)
                    aug_features1edge = aug_features1edge.to(self.device)
                    aug_features2edge = aug_features2edge.to(self.device)
                    adj = adj.to(self.device)
                    aug_adj1edge = aug_adj1edge.to(self.device)
                    aug_adj2edge = aug_adj2edge.to(self.device)
                b_xent = nn.BCEWithLogitsLoss()
                xent = nn.CrossEntropyLoss()
                self.train()
                optimiser.zero_grad()
                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]
                lbl_1 = torch.ones(1, nb_nodes)
                lbl_2 = torch.zeros(1, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)
                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.to(self.device)
                    lbl = lbl.to(self.device)
                logit = self(features, shuf_fts, aug_features1edge, aug_features2edge,
                            adj,
                            aug_adj1edge,
                            aug_adj2edge,
                            False, None, None, None, lbl=lbl,sample=negetive_sample)
                loss = loss + logit
                # print(loss)
                showloss = loss/(step+1)
            loss = loss / (step+1)
            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, 1000, loss.item()))
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                # torch.save(model.state_dict(), args.save_name)
            else:
                cnt_wait += 1
                # print("cnt_wait",cnt_wait)

            if cnt_wait == patience:
                print('-' * 100)
                print('Early stopping at '+str(epoch) +' eopch!')
                break

        folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.state_dict(),
                    "./Experiment/pre_trained_model/{}/{}.pth".format(self.dataset_name, 'MultiGprompt'))
        print("+++model saved ! {}/{}.pth".format(self.dataset_name, 'MultiGprompt'))


def mygather(feature, index): 
    input_size=index.size(0)
    index = index.flatten() # 2485 * 201 = torch.Size([499485])
    index = index.reshape(len(index), 1) # torch.Size([499485, 1])
    # 扩展维度 
    # example 
    # x = torch.tensor([[1],
    #               [2],
    #               [3]])
    # torch.broadcast_to(x, (3, 3))
    # tensor([[1, 1, 1],
    #     [2, 2, 2],
    #     [3, 3, 3]])
    index = torch.broadcast_to(index, (len(index), feature.size(1))) # torch.Size([499485, 256])
    # res得到499485个索引对应的feature
    res = torch.gather(feature, dim=0, index=index) # torch.Size([499485, 256])
    # example 
    # a = tensor([
    #     [ 0,  1,  2,  3,  4],
    #     [ 5,  6,  7,  8,  9],
    #     [10, 11, 12, 13, 14]])

    # 张量b的元素都是对应张量a的索引
    # b = tensor(
    # [[1, 0, 0, 0, 0],
    # [0, 0, 1, 0, 0],
    # [0, 0, 0, 0, 0]])

    # c = a.gather(0, b) # dim=0  因为dim=0，c的第一列的值[5,0,0], 从a的第一列[0,5,10]取对应b中第一列的索引[1,0,0]得到
    # d = a.gather(1, b) # dim=1  因为dim=1，d的第一列的值[1,5,10],从a的每一行对应的列索引[1,0,0]得到，即a[0][1],a[1][0],a[2][0]
    # c= tensor([
    #         [5, 1, 2, 3, 4],
    #         [0, 1, 7, 3, 4],
    #         [0, 1, 2, 3, 4]])
    # d=tensor([
    #         [ 1,  0,  0,  0,  0],
    #         [ 5,  5,  6,  5,  5],
    #         [10, 10, 10, 10, 10]])

    # 通过reshape得到每一个节点对应的所有正负样本的feature torch.Size([2485, 201, 256])
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature,device):
    # feature.shape  torch.Size([2485, 256])
    # tuples.shape   torch.Size([2485, 201])
    h_tuples=mygather(feature,tuples) # torch.Size([2485, 201, 256])
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1) # torch.Size([2485, 1])
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp=temp.to(device)
    h_i = mygather(feature, temp) #torch.Size([2485, 201, 256])
    sim = F.cosine_similarity(h_i, h_tuples, dim=2) # torch.Size([2485, 201])
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0) # torch.Size([201, 2485])
    numerator = exp[0].reshape(-1, 1) # torch.Size([2485, 1]) 每个节点对应的一个正样本的相似度
    denominator = exp[1:exp.size(0)] 
    denominator = denominator.permute(1, 0) # torch.Size([2485, 200]) 每个节点对应的200个负样本的相似度
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator) # 更靠近正样本而原理负样本
    return res.mean()


def prompt_pretrain_sample(adj,n):
    # csr矩阵是压缩稀疏行格式
    # indptr 中的值可以这样理解，通过第 i + 1 行和第 i 行值的差表示第 i 行拥有多少非0元素，也可以是一种长度的指示，用来从induces中寻找确切的非零值列号
    # 对于第 i 行而言，该行中非零元素的列索引为 indices[indptr[i]:indptr[i+1]]
    # 得到了行索引、列索引，相应的数据存放在： data[indptr[i]:indptr[i+1]]
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling disconnected tuples")
    # 加了个trange，对抓取添加进度条能更加直观的看到生成了多少，就是进度条可视化
    for i in trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        # np.setdiff1d找到2个数组中集合元素的差异，返回在whole中但不在nonzero_index_i_row中的已排序的唯一值。
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        # 从nonzero中（即相邻的节点）取一个，即一个正样本，从zero中（即不相邻的节点）取n个，即n个负样本
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

def tu_prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

class weighted_feature(nn.Module):
    def __init__(self,a1,a2,a3):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,3), requires_grad=True)
        self.reset_parameters(a1,a2,a3)
    def reset_parameters(self,a1,a2,a3):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(a1)
        self.weight[0][1].data.fill_(a2)
        self.weight[0][2].data.fill_(a3)
    def forward(self, graph_embedding1,graph_embedding2,graph_embedding3):
        print("weight",self.weight)
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2 + self.weight[0][2] * graph_embedding3
        return graph_embedding