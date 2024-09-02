import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from prompt_graph.utils import act,cmd
from deprecated.sphinx import deprecated
from sklearn.cluster import KMeans
from torch_geometric.nn.inits import glorot
from kmeans_pytorch import kmeans 
from sklearn.cluster import KMeans
import numpy as np

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_robust_pg, num_robust_pg, inner_prune=None):
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
        self.num_robust_pg = num_robust_pg

        self.inner_prune = inner_prune
        self.robust_token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_robust_pg, token_dim)) for i in range(num_robust_pg)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.robust_token_list:
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
        for i, tokens in enumerate(self.robust_token_list):
            # pg_list.append(tokens)

            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()
            # 这里pg的y其实是没有用的，只是一个不同prompt图的标识，能不能考虑每个类生成一个prompt
            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        # pg_batch = Batch.from_data_list(pg_list)
        pg_batch = pg_list
        return pg_batch

class RobustPrompt_I(LightPrompt):
    def __init__(self, token_dim, per_graph_token_num, num_prompt_graph = 1,cross_prune=0.1, inner_prune=0.01):
        super(RobustPrompt_I, self).__init__(token_dim, per_graph_token_num, num_prompt_graph, inner_prune)  # only has one prompt graph.
        self.num_prompt_graph      = num_prompt_graph
        self.per_graph_token_num   = per_graph_token_num
        self.cross_prune           = cross_prune
        # self.token_num             = token_num


    def add_robust_prompt(self, graph_batch: Batch):
        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        robust_prompt_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num # 相当于每一个节点的编号+token_num,把前面的编号留给pg图

            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < -1, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            # edge_index = torch.cat([g_edge_index, cross_edge_index], dim=1)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            robust_prompt_graph_list.append(data)

        robust_graphp_batch = Batch.from_data_list(robust_prompt_graph_list)
        return robust_graphp_batch



    def forward(self, graph_batch, pseudo_model):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)
        # inner_edge_index = pg.edge_index
        # token_num = pg.x.shape[0]

        # pg = self.robust_token_list
        self.total_token_num = self.num_prompt_graph * self.per_graph_token_num

        num_nodes_prompt_graphs  = []
        num_nodes_induced_graphs = []
        re_graph_list = []

        pruned_graph_list = []
        prompt_graph_list = []

        for g in Batch.to_data_list(graph_batch):

            # 我们假定预训练过程中用到的数据集是干净的，但是在下游任务上的图是被攻击或扰动的，因此可以认为预训练得到的模型在干净图上具有很好的效果


            # # 得到根据相似度过滤得到修剪图，我们认为预训练后的模型在过滤图上的结果是可靠的
            # ##################################################
            # # Prune edge index
            # edge_index = g.edge_index
            # cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            # # Define threshold t
            # threshold = 0.1
            # # Identify edges to keep
            # keep_edges = cosine_sim >= threshold
            # # Filter edge_index to only keep edges above the threshold
            # pruned_edge_index = edge_index[:, keep_edges]
            # pruned_g          = Data(x=g.x, edge_index=pruned_edge_index)
            # # 这里后面返回两个batch，一个batch是filtered graph batch，另一个batch是prompt batch
            # pruned_graph_list.append(pruned_g)
            # # get filtered graph batch 
            # ##################################################





            # # 获得在被攻击后的图上添加prompt的图（通过和过滤图输出结果的逼近从而学习能进行鲁棒prompt的图）
            # ##################################################
            # g_edge_index = g.edge_index + token_num # 相当于每一个节点的编号+token_num,把前面的编号留给pg图
            # # 这里感觉可以做一个可视化的case study的实验，一开始的图同质性假设不好，加上我们的提示图之后，同质性假设变小。
            # # 计算输入图和pg图之间的关系 如果只有一个图，不用计算图内的关系，如果每个分类都有一个图，则计算每个图内部的链接关系加入进去。相似度计算，torch.cosine
            # cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            # cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            # # cross_sim = torch.cosine_similarity(pg.unsqueeze(1), g.x.unsqueeze(0), dim=-1)
            # cross_adj = torch.where(cross_sim < -1, 0, cross_sim)
            # cross_edge_index = cross_adj.nonzero().t().contiguous()
            # cross_edge_index[1] = cross_edge_index[1] + token_num
            # # print(cross_edge_index.shape)
            # # print(inner_edge_index.shape)
            # x = torch.cat([pg.x, g.x], dim=0)
            # y = g.y
            # # edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            # edge_index = torch.cat([g_edge_index, cross_edge_index], dim=1)
            # prompt_g = Data(x=x, edge_index=edge_index, y=y)
            # prompt_graph_list.append(prompt_g)
            # ##################################################




            # 获得在被攻击后的图上添加prompt的图，每个图先根据特征做一个聚类，对节点进行一个初分类，然后对于每一个类都添加一个prompt图
            ##################################################
            # g.relabel_central_index
            cluster_y_pred = torch.argmax(pseudo_model(g.x), dim=1)
            print(cluster_y_pred[g.relabel_central_index])
            print(g.y)
            quit()
            # cluster_y_pred, cluster_centers = kmeans(X=g.x, num_clusters=self.num_prompt_graph, distance='euclidean', device=torch.device('cuda')) # 手动关闭了tqdm
            # cluster_y_pred = torch.tensor(KMeans(n_clusters=self.num_prompt_graph, random_state=0).fit_predict(g.x.detach().cpu()))
            # cluster_y_pred = torch.ones(g.x.shape[0])


            g_edge_index = g.edge_index + self.total_token_num # 相当于每一个节点的编号+token_num,把前面的编号留给pg图

            # print(self.per_graph_token_num)
            # print(total_token_num)

            total_cross_edge_index = torch.tensor([]).cuda()
            total_inner_edge_index = torch.tensor([]).cuda()
            total_x                = torch.tensor([]).cuda()
            cur_index = 0 # 用于判断

            for pseudo_labels in range(self.num_prompt_graph):
                
                pseudo_index = torch.nonzero(cluster_y_pred == pseudo_labels).squeeze(-1) # 注意 .squeeze() 是把所有为 1 的维度删掉,所以加-1只删最后一维
                # 伪标签存在才可以加提示，在相同伪标签的节点中添加提示
                if len(pseudo_index) != 0:
                    # 每个prompt graph内部相连
                    pseudo_inner_edge_index = pg[pseudo_labels].edge_index + self.per_graph_token_num * pseudo_labels
                    # 每个prompt graph和具有特定伪标签的节点相链接
                    pseudo_cross_dot = torch.mm(pg[pseudo_labels].x, torch.transpose(g.x[pseudo_index], 0, 1))
                    pseudo_cross_sim = torch.sigmoid(pseudo_cross_dot)  # 0-1 from prompt to input graph
                    pseudo_cross_adj = torch.where(pseudo_cross_sim < self.cross_prune, 0, pseudo_cross_sim)
                    pseudo_cross_edge_index = pseudo_cross_adj.nonzero().t().contiguous()
                    # 每个prompt_graph的index要按长度更新，而不是统一更新
                    pseudo_cross_edge_index[0] = pseudo_cross_edge_index[0] + self.per_graph_token_num * pseudo_labels
                    # 原始图的index进行统一的total更新
                    pseudo_cross_edge_index[1] = pseudo_cross_edge_index[1] + self.total_token_num

                    # 结合
                    total_cross_edge_index = torch.cat([total_cross_edge_index, pseudo_cross_edge_index], dim = 1)
                    total_inner_edge_index = torch.cat([total_inner_edge_index, pseudo_inner_edge_index], dim = 1)
                    
                # 进行一个判断，这里如果图中存在不存在伪标签的节点，就不增加连接关系，而是以孤立节点的形式存在，当然也可以设计成如何辅助其他的提示图
                total_x = torch.cat([total_x, pg[pseudo_labels].x], dim = 0)
                assert torch.equal(total_x[cur_index: cur_index + self.per_graph_token_num], pg[pseudo_labels].x)
                cur_index += self.per_graph_token_num
                
            total_x = torch.cat([total_x, g.x], dim = 0)
            y = g.y

            edge_index = torch.cat([total_inner_edge_index.long(), g_edge_index, total_cross_edge_index.long()], dim=1)
            prompt_g = Data(x=total_x, edge_index=edge_index, y=y)
            re_graph_list.append(prompt_g)
            
            num_nodes_induced_graphs.append(g.num_nodes)
            num_nodes_prompt_graphs.append(prompt_g.num_nodes)

            # 记录每个添加prompt后图的节点数量
            ##################################################
            # num_nodes_prompt_graphs.append(prompt_g.num_nodes)
            # num_nodes_prompt_graphs.append(total_token_num)
            ##################################################
        # pruned_graph_batch = Batch.from_data_list(pruned_graph_list)
        # prompt_graph_batch = Batch.from_data_list(prompt_graph_list)
        # return pruned_graph_batch, prompt_graph_batch, num_nodes_prompt_graphs

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch, num_nodes_induced_graphs, num_nodes_prompt_graphs
        
    

    def Tune(self, train_loader, remaining_loader, pseudo_model, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader): 
            train_batch = train_batch.to(device)
            
            #######################################################################################################################
            # idea 1
            prompted_graph, num_nodes_induced_graphs, num_nodes_prompt_graphs = self.forward(train_batch, pseudo_model)
            # print(num_nodes_induced_graphs)
            # print(num_nodes_prompt_graphs)
            # print(prompted_graph)
            # print(prompted_graph.x.shape)

            prompt_node_emb, prompt_graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch, prompt_type = 'RobustPrompt_I')       


            prompt_node_emb_token = prompt_node_emb[0: self.total_token_num]
            # 保存prompt graph上除了token的所有节点的embedding
            start = num_nodes_prompt_graphs[0]
            for lenPG in num_nodes_prompt_graphs[1:]:
                prompt_node_emb_token = torch.cat((prompt_node_emb_token, prompt_node_emb[start: start + self.total_token_num]), dim = 0)
                start += lenPG 

            # print(prompt_node_emb_token.shape)
            loss_mse = F.mse_loss(prompt_node_emb[prompted_graph.edge_index[0]], prompt_node_emb[prompted_graph.edge_index[1]])
            # print(loss_mse)
            pre = answering(prompt_graph_emb)
            train_loss = lossfn(pre, train_batch.y) + loss_mse
         
            #######################################################################################################################


            #######################################################################################################################
            # idea 2
            # pruned_graph, prompt_graph, num_nodes_prompt_graphs = self.forward(train_batch)
            # 方法一，让修建图和提示图输出embedding相同
            # inner loss : Node embedding alignment.
            # 得到pruned graph的emb和prompt graph的emb
            # pruned_node_emb, pruned_graph_emb = gnn(pruned_graph.x, pruned_graph.edge_index, pruned_graph.batch, prompt_type = 'RobustPrompt_I')
            # prompt_node_emb, prompt_graph_emb = gnn(prompt_graph.x, prompt_graph.edge_index, prompt_graph.batch, prompt_type = 'RobustPrompt_I')           
            # prompt_node_emb_wo_token = prompt_node_emb[self.token_num: num_nodes_prompt_graphs[0]]
            # # 保存prompt graph上除了token的所有节点的embedding
            # start = num_nodes_prompt_graphs[0]
            # for lenPG in num_nodes_prompt_graphs[1:]:
            #     prompt_node_emb_wo_token = torch.cat((prompt_node_emb_wo_token, prompt_node_emb[start + self.token_num: start + lenPG]), dim = 0)
            #     start += lenPG 
            
            # loss_mse = F.mse_loss(prompt_node_emb_wo_token, pruned_node_emb)
            # print("mse loss : {}".format(loss_mse))
            # mse的实现方式
            # loss_mse = (prompt_node_emb_wo_token - pruned_node_emb).pow(2).sum(1).sum()
            # print(loss_mse / (pruned_node_emb.shape[0] * pruned_node_emb.shape[1]))




            # 方法二，需要对prompt token进行约束，让提示图具有比较好的同质性
            # loss_homo = F.mse_loss(prompt_node_emb[prompt_graph.edge_index[0]], prompt_node_emb[prompt_graph.edge_index[1]])
            # print("homo loss : {}".format(loss_homo))




            # 方法三，分布loss，distribution loss:  graph embedding distribution alignment.
            # remaing_graph_embs = torch.tensor([]).to(device)
            # for batch_id, remaining_batch in enumerate(remaining_loader):  
            #     remaining_batch = remaining_batch.to(device)
            #     remaining_prompt_graph_batch = self.add_robust_prompt(remaining_batch)
            #     _, remaining_prompt_graph_batch_emb = gnn(remaining_prompt_graph_batch.x, remaining_prompt_graph_batch.edge_index, remaining_prompt_graph_batch.batch, prompt_type = 'RobustPrompt')
            #     remaing_graph_embs = torch.cat((remaing_graph_embs, remaining_prompt_graph_batch_emb),dim = 0)
            # cluster = KMeans(n_clusters=7,random_state=0).fit(remaing_graph_embs.detach().cpu())
            # remaing_embs_dis = torch.FloatTensor(cluster.cluster_centers_).to(device)
            # loss_cmd = cmd(remaing_embs_dis, prompt_graph_emb)
            # print("cmd loss : {}".format(loss_cmd))

            # pre = answering(prompt_graph_emb)
            # train_loss = lossfn(pre, train_batch.y) +  loss_homo + loss_cmd
            #######################################################################################################################


            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
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
