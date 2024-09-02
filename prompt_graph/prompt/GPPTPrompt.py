import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleMeanConv(MessagePassing):
    def __init__(self):
        # 初始化时指定聚合方式为 'mean'，即平均聚合
        super(SimpleMeanConv, self).__init__(aggr='mean')  # 'mean'聚合。

    def forward(self, x, edge_index):
        # x 代表节点特征矩阵，edge_index 是图的边索引列表

        # 在边索引中添加自环，这样在聚合时，节点也会考虑自己的特征
        # 如果有自环，就不需要重复添加了，直接用
        from torch_geometric.utils import contains_self_loops
        if not contains_self_loops(edge_index):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 开始消息传递过程，其中x是每个节点的特征，edge_index定义了节点间的连接关系
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j 表示邻居节点的特征，这里直接返回，因为我们使用的是 'mean' 聚合
        return x_j

class GPPTPrompt(torch.nn.Module):
    def __init__(self, n_hidden, center_num, n_classes, device):
        super(GPPTPrompt, self).__init__()
        self.center_num = center_num
        self.n_classes = n_classes
        self.device = device
        self.StructureToken = torch.nn.Linear(n_hidden, center_num, bias=False)
        self.StructureToken=self.StructureToken.to(device)  # structure token
        self.TaskToken = torch.nn.ModuleList()
        for i in range(center_num):
            # 每个TaskToken都是一个独立的包含不同标签的token，对应后面的正交约束
            self.TaskToken.append(torch.nn.Linear(2 * n_hidden, n_classes, bias=False))  
            # 这里初始化是2倍的n_hidden，但是在下面用均值后的值初始化变成了单倍的n_hidden,GPPT源码是把向量拼接在一起，但是源码的实现有问题，所以这里直接用n_hidden也可以
        
        #task token
        self.TaskToken = self.TaskToken.to(device)

    def weigth_init(self, h, edge_index, label, index):
        # 对于图中的每一个节点，将其特征（'h'）发送给所有邻居节点，然后每个节点会计算所有收到的邻居特征的平均值，并将这个平均值存储为自己的新特征在'neighbor'下

        conv = SimpleMeanConv()
        # 使用这个层进行前向传播，得到聚合后的节点特征
        h = conv(h, edge_index)

        # GPPT源码是把向量拼接在一起
        # h_neighbor = conv(h, edge_index)
        # h = torch.cat((h, h_neighbor),dim=1)

        features=h[index]
        labels=label[index.long()]

        # 对train set做聚类，如果1 shot则每一个节点一个类， features的shape只有train
        cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(features.detach().cpu())
        temp=torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.StructureToken.weight.data = temp.clone().detach()

        p=[]
        for i in range(self.n_classes):
            p.append(features[labels==i].mean(dim=0).view(1,-1))
        temp=torch.cat(p,dim=0).to(self.device)
        for i in range(self.center_num):
            # 这里本来是2倍的n_hidden,用均值后的值初始化变成了单倍的n_hidden，初始化每个token都一样
            self.TaskToken[i].weight.data = temp.clone().detach()
        
    
    def update_StructureToken_weight(self, h):
        if h.size(0)>20000:
            cluster_ids_x, cluster_centers = KMeans(X=h, num_clusters=self.center_num, distance='euclidean', device=self.device)
            self.StructureToken.weight.data = cluster_centers.clone()
        else:
            cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(h.detach().cpu())
            temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
            self.StructureToken.weight.data = temp.clone()


    def get_TaskToken(self):
        pros=[]
        for name,param in self.named_parameters():
            if name.startswith('TaskToken.'):
                pros.append(param)
        return pros
        
    def get_StructureToken(self):
        for name,param in self.named_parameters():
            if name.startswith('StructureToken.weight'):
                pro=param
        return pro
    
    def get_mid_h(self):
        return self.fea


    def forward(self, h, edge_index):
        device = h.device
        conv = SimpleMeanConv()
        # 使用这个层进行前向传播，得到聚合后的节点特征
        h = conv(h, edge_index)
        self.fea = h 

        out = self.StructureToken(h)
        index = torch.argmax(out, dim=1)
        out = torch.FloatTensor(h.shape[0],self.n_classes).to(device)
        for i in range(self.center_num):
            out[index==i]=self.TaskToken[i](h[index==i])
        return out
    
