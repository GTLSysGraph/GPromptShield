import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from prompt_graph.utils import act
from deprecated.sphinx import deprecated
from sklearn.cluster import KMeans
from torch_geometric.nn.inits import glorot

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1)) # 例如，token的特征维度是[10, 1433]
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous() 

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long())) # 构建了10个节点的全连接提示图

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch

class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num 
            # 相当于每一个节点的编号+token_num,把前面的编号留给prompt图，这个prompt图就是一个10个节点组成的全链接图, 例如，token的特征维度是[10, 1433], 就弄了一个10个节点的全连接可训练图当prompt图

            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))  # 这个是这10个节点的prompt图和Cora图上所有节点连接，比如cora 2485个节点，这个就是[10，2485]的矩阵
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim) # 
            
            cross_edge_index = cross_adj.nonzero().t().contiguous() # 对这个[10，2485]的矩阵修剪了一下当作token和Cora的连接提示图
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)   # 把prompt图的token和cora的特征拼接在一起
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1) # inner 就是prompt图的内部自己连接，g_edge_index就是cora自己的边，cross_edge_index就是prompt和cora的连接
            data = Data(x=x, edge_index=edge_index, y=y) 
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch
    

    def Tune(self, train_loader, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):  

            # ################
            # pruned_batch_list = []
            # for g in Batch.to_data_list(train_batch):
            #       # Prune edge index
            #       edge_index = g.edge_index
            #       cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            #       # Define threshold t
            #       threshold = 0.1
            #       # Identify edges to keep
            #       keep_edges = cosine_sim >= threshold
            #       # Filter edge_index to only keep edges above the threshold
            #       pruned_edge_index = edge_index[:, keep_edges]
            #       pruned_g          = Data(x=g.x, edge_index=pruned_edge_index,y=g.y, relabel_central_index= g.relabel_central_index, raw_index = g.raw_index, pseudo_label= g.pseudo_label)
            #       pruned_batch_list.append(pruned_g)
            # train_batch = Batch.from_data_list(pruned_batch_list)
            # ################

            # ###############
            # pruned_batch_list = []
            # for g in Batch.to_data_list(train_batch):
            #     g = g.to(device)
            #     logits_ptb = gnn(g.x, g.edge_index)
            #     logits_ptb = torch.concat((logits_ptb, g.x), dim=1)
            #     features_edge = torch.concat((logits_ptb[g.edge_index[0]], logits_ptb[g.edge_index[1]]), dim=1)
            #     remove_flag = torch.zeros(g.edge_index.shape[1], dtype=torch.bool).to(device)
            #     for k in range(len(detectors)):
            #         output = F.sigmoid(detectors[k](features_edge)).squeeze(-1)
            #         remove_flag = torch.where(output > 0.1, True, remove_flag)
            #     keep_edges = remove_flag == False
            #     pruned_edge_index = g.edge_index[:, keep_edges]
            #     pruned_g          = Data(x=g.x, edge_index=pruned_edge_index,y=g.y, relabel_central_index= g.relabel_central_index, raw_index = g.raw_index, pseudo_label= g.pseudo_label)
            #     pruned_batch_list.append(pruned_g)
            # train_batch = Batch.from_data_list(pruned_batch_list)
            # ###############


            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, train_batch.y)
            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
        return running_loss / len(train_loader)
    

    # 分开更新prompt和answer没有效果，感觉应该还是要一起调整才可以
    def TuneKnowledgeDistillation(self, train_loader, pseudo_logits_train, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):  
            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)

            # print(train_batch.y)
            # print(torch.argmax(pseudo_logits_train, dim=1))
            # true_predict = train_batch.y == torch.argmax(pseudo_logits_train, dim=1)
            # print(true_predict)
            # print(sum(true_predict))
            # quit()

            # loss_ce = lossfn(pre, train_batch.y)
            loss_ce = lossfn(pre, torch.argmax(pseudo_logits_train, dim=1))
            
            # KL散度，知识蒸馏
            temperature = 1.0
            alpha = 0.8
            pseudo_logits_train = pseudo_logits_train.detach()
            loss_kl = torch.nn.KLDivLoss()(F.log_softmax(pre / temperature, dim=1), F.softmax(pseudo_logits_train / temperature, dim=1)) 
            loss = (1 - alpha) * loss_ce + alpha * loss_kl

            opi.zero_grad()
            loss.backward()
            opi.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)



    def TuneWithoutAnswering(self, train_loader, gnn, answering, lossfn, opi, device):
        total_loss = 0.0 
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            emb0 = gnn(batch.x, batch.edge_index, batch.batch)
            pg_batch = self.inner_structure_update()
            pg_batch = pg_batch.to(self.device)
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            # cross link between prompt and input graphs
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            sim = torch.softmax(dot, dim=1)
            loss = lossfn(sim, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()  
        return total_loss / len(train_loader) 


class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre


