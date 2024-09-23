import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree
import numpy as np

# Done! next step
class RobustPrompt_I(torch.nn.Module):
    def __init__(self, in_channels: int, muti_defense_pt_dict, use_attention, num_heads, kl_global, cosine_constraint, pt_threshold, temperature, weight_mse, weight_kl, weight_constraint):
        super(RobustPrompt_I, self).__init__()

        self.in_channels       = in_channels
        self.pt_dict           = muti_defense_pt_dict
        self.pt_keys           = self.pt_dict.keys()

        self.use_attention     = use_attention       # 是否使用attention融合不同prompt
        self.kl_global         = kl_global           # kl散度是全局还是局部
        self.cosine_constraint = cosine_constraint   # 是否用cosine计算prompt间距离
        self.pt_threshold      = pt_threshold        # 添加final prompt后的修剪超参数

        # Tune过程中的不同loss权重和temperature
        self.temperature       = temperature
        self.weight_mse        = weight_mse
        self.weight_kl         = weight_kl
        self.weight_constraint = weight_constraint

        
        print('use RobustPrompt Inductive')
        print('defense pt dict : ', self.pt_dict)
        print('use_attention : ',self.use_attention)
        print('kl_global : ',self.kl_global)
        print('cosine_constraint : ',self.cosine_constraint)
        print('pt_threshold : ',self.pt_threshold)
        print('temperature : ',self.temperature)
        print('weight_mse',self.weight_mse)
        print('weight_kl : ',self.weight_kl)
        print('weight_constraint : ',self.weight_constraint)



        if 'sim_pt' in self.pt_keys:
            self.prompt_sim_pt              = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        if 'degree_pt' in self.pt_keys:
            self.prompt_degree_pt           = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 
        if 'out_detect_pt' in self.pt_keys:
            self.prompt_out_detect_pt       = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 
        if 'other_pt' in self.pt_keys:
            self.prompt_other_pt            = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 


        # attention  注意要用batch_first 因为nlp默认的是batch_first=False (L,N,Eq) L为目标序列长度，N为批量大小，Eq为查询嵌入维数embed_dim，batch_first=True时(N,L,Eq)
        # 目前head仅支持1，参数太多会过拟合
        assert num_heads == 1
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim = self.in_channels * num_heads, num_heads = num_heads, dropout = 0.0, batch_first=True)
        self.readout_token   = torch.nn.Parameter(torch.randn(1, 1, self.in_channels))
        self.reset_parameters()


    def reset_parameters(self):
        if 'sim_pt' in self.pt_keys:
            glorot(self.prompt_sim_pt)
        if 'degree_pt' in self.pt_keys:
            glorot(self.prompt_degree_pt)
        if 'out_detect_pt' in self.pt_keys:
            glorot(self.prompt_out_detect_pt) 
        if 'other_pt' in self.pt_keys:
            glorot(self.prompt_other_pt)
        glorot(self.readout_token)



    def add_pt(self, x: torch.Tensor, specified_prompt):
        return x + specified_prompt  



    def get_muti_prompt(self, node_use_each_pt_whole_batch, device):
        muti_prompt   = []
        overlap_list  = []
        for name, param in self.named_parameters():
            if name.startswith('prompt'):    
                muti_prompt.append(param) # shape torch.Size([1, 1433])
                node_use_each_pt = node_use_each_pt_whole_batch[name.split('_', 1)[1]]
                # 这里是判断是 tensor还是list 因为用self.kl_global, 全局做kl是tensor，局部做kl得到的是list
                if not self.kl_global:
                    node_use_each_pt =  torch.tensor(np.concatenate(node_use_each_pt)).to(device)
                overlap_list.append(node_use_each_pt) # example prompt[0] _ sim_pt[1] -> node_use_each_pt_whole_batch['sim_pt']

        muti_prompt = torch.cat(muti_prompt, dim = 0) # 注意！stack和cat不一样，如果有两个形状为(a, b)的张量，用torch.stack()在第一个维度上将它们堆叠成一个形状为(2, a, b)的新张量，会增加一个维度,这里用cat就可以，如果用stack需要squeeze(1)，压缩中间的维度

        overlap_matrix = torch.eye(len(overlap_list)).to(device)
        for i in range(len(overlap_list)):
            for j in range(i + 1, len(overlap_list)):
                # 两个prompt使用的节点交集 两种方式结果相同
                mask = torch.isin(overlap_list[i], overlap_list[j]) # mask = torch.isin(overlap_list[j], overlap_list[i])
                overlap_nodes = overlap_list[i][mask]               # overlap_nodes = overlap_list[j][mask]
                #  两个prompt使用的节点并集
                union_nodes = torch.unique(torch.cat((overlap_list[i],overlap_list[j])))
                overlap_matrix[i][j] = len(overlap_nodes) / len(union_nodes)
                overlap_matrix[j][i] = len(overlap_nodes) / len(union_nodes)
        return muti_prompt, overlap_matrix




    def forward(self, graph_batch: Batch, device):
        # 记录每个batch中所有图在每个prompt上用到的节点, 同时记录一个start index，在每个图处理完后加上图的节点数量以作为下一个图的开始索引
        whole_batch_start_index = 0
        node_use_each_pt_whole_batch = {}
        for pt in self.pt_keys:
            if self.kl_global:
                node_use_each_pt_whole_batch[pt] = torch.tensor([]).to(device)     # 用tensor直接把所有batch的都保存
            else:
                node_use_each_pt_whole_batch[pt] = []                              # 用list可以分开存每一个图的pt使用的节点
        node_use_each_pt_whole_batch['g_start_index'] = []

        graph_mutiftpt = []
        for g in Batch.to_data_list(graph_batch):
            # 首先用0.初始化并拼接所有的pt长度
            g_mutiftpt_record = torch.zeros(g.num_nodes, len(self.pt_keys) * self.in_channels)
            g_mutiftpt_record = g_mutiftpt_record.to(device)
            
            # 记录拼接所有pt的整体向量中每个pt的位置
            pt_range_dict = {}
            range_start = 0
            for pt in self.pt_keys:
                pt_range_dict[pt] = [range_start, range_start + self.in_channels]
                range_start += self.in_channels
            # {'sim_pt': [0, 1433], 'degree_pt': [1433, 2866], 'other_pt': [2866, 4299]}

            x = g.x
            edge_index = g.edge_index
            # 用于记录当前图中所有defense pt用到的节点
            node_use_pt = torch.tensor([]).to(device)


            if 'sim_pt' in self.pt_keys:
                # print('sim_pt : ',self.pt_dict['sim_pt'])
                # 相似度(cos (θ))值范围从-1(不相似)到+1(非常相似) nan代表孤立节点，对结果不造成影响
                x_norm = x / torch.sqrt(torch.sum(x * x,dim=1)).unsqueeze(-1)
                e = torch.sum(x_norm[edge_index[0]] * x_norm[edge_index[1]], dim = 1).unsqueeze(-1)
                row, col = edge_index
                c = torch.zeros(x_norm.shape[0], 1).to(device)
                c = c.scatter_add_(dim=0, index=col.unsqueeze(1), src=e)
                deg = degree(col, x.size(0), dtype=x.dtype).unsqueeze(-1) 
                csim = c / deg
                csim = csim.squeeze()
                node_use_sim_pt = torch.nonzero(csim <= self.pt_dict['sim_pt']).squeeze(-1) # 不能直接用squeeze()，会把所有1维度都压缩，当只有单个节点会有问题
  

                # 记录当前图中用到sim pt的node 局部的角度，为了后面筛选出一个图中没有用到任何defense pt的node
                node_use_pt = torch.concat((node_use_pt, node_use_sim_pt))
                # 将prompt放到当前图中用sim pt的指定节点上
                g_mutiftpt_record[node_use_sim_pt, pt_range_dict['sim_pt'][0] : pt_range_dict['sim_pt'][1]] = self.prompt_sim_pt
                # 记录每个图用到sim pt的node   全局的角度  注意这里一定要放在对当前图处理完后面，要不会改变节点索引
                node_use_sim_pt = node_use_sim_pt + whole_batch_start_index # 从当前图的start index进行记录  注意！不要用+=，会出现无法计算梯度问题
                if self.kl_global:
                    node_use_each_pt_whole_batch['sim_pt'] = torch.concat((node_use_each_pt_whole_batch['sim_pt'], node_use_sim_pt))
                else:
                    node_use_each_pt_whole_batch['sim_pt'].append(node_use_sim_pt.tolist())


            if 'degree_pt' in self.pt_keys:
                # print('degree_pt : ',self.pt_dict['degree_pt'])
                deg = degree(col, x.size(0), dtype=x.dtype)
                node_use_degree_pt = torch.nonzero(deg <= self.pt_dict['degree_pt']).squeeze(-1) # 不能直接用squeeze()，会把所有1维度都压缩，当只有单个节点会有问题

                # 记录当前图中用到degree pt的node 局部的角度，为了后面筛选出一个图中没有用到任何defense pt的node
                node_use_pt = torch.concat((node_use_pt, node_use_degree_pt))
                # 将prompt放到当前图中用degree pt的指定节点上
                g_mutiftpt_record[node_use_degree_pt, pt_range_dict['degree_pt'][0] : pt_range_dict['degree_pt'][1]] = self.prompt_degree_pt
                # 记录每个图用到degree pt的node 全局的角度  注意这里一定要放在对当前图处理完后面，要不会改变节点索引
                node_use_degree_pt = node_use_degree_pt + whole_batch_start_index # 从当前图的start index进行记录  注意！不要用+=，会出现无法计算梯度问题
                if self.kl_global:
                    node_use_each_pt_whole_batch['degree_pt'] = torch.concat((node_use_each_pt_whole_batch['degree_pt'],  node_use_degree_pt))
                else:
                    node_use_each_pt_whole_batch['degree_pt'].append(node_use_degree_pt.tolist())


            if 'out_detect_pt' in self.pt_keys:
                pass


            if 'other_pt' in self.pt_keys:
                # print('use other_pt, tips: Apply to the remaining nodes without adding pt! IF only other_pt,equal GPF')
                all_nodes    = torch.arange(0, g.num_nodes).to(device)
                all_pt_nodes = torch.unique(node_use_pt)
                mask = ~torch.isin(all_nodes, all_pt_nodes)
                node_use_no_pt = all_nodes[mask]
                # 对other节点的选择方式，包括选择所有other节点的'all'方式和从中随机选择一些节点的'random-0.2'方法
                if len(node_use_no_pt) > 0:
                    if self.pt_dict['other_pt'] == 'all':
                        node_use_other_pt = node_use_no_pt
                    elif self.pt_dict['other_pt'].split('-')[0] == 'random':
                        num_samples     = int(float(self.pt_dict['other_pt'].split('-')[1]) * node_use_no_pt.size(0))
                        random_indices  = torch.randint(0, node_use_no_pt.size(0), (num_samples,))
                        node_use_other_pt  = node_use_no_pt[random_indices]
                else:
                    node_use_other_pt = torch.tensor([], dtype=node_use_no_pt.dtype).to(device)

                # 将prompt放到当前图中没有用到任何defense pt的other节点上
                g_mutiftpt_record[node_use_other_pt, pt_range_dict['other_pt'][0] : pt_range_dict['other_pt'][1]] = self.prompt_other_pt
                # 记录每个图没有用到任何defense pt但添加了other_pt的node 全局的角度 注意这里一定要放在对当前图处理完后面，要不会改变节点索引
                node_use_other_pt = node_use_other_pt + whole_batch_start_index # 从当前图的start index进行记录  注意！不要用+=，会出现无法计算梯度问题
                if self.kl_global:
                    node_use_each_pt_whole_batch['other_pt'] = torch.concat((node_use_each_pt_whole_batch['other_pt'], node_use_other_pt))
                else:
                    node_use_each_pt_whole_batch['other_pt'].append(node_use_other_pt.tolist())
    


            g_mutiftpt_record = g_mutiftpt_record.reshape(g.num_nodes, len(self.pt_keys), self.in_channels)
            ######################################################################################
            if self.use_attention:
                # 用self-attention
                # 加一个readout_token
                g_mutiftpt_record = torch.cat([self.readout_token.expand(g.num_nodes, 1, self.in_channels), g_mutiftpt_record], dim=1)
                # padding位置
                padding = torch.zeros(self.in_channels).to(device)
                key_padding_mask = torch.all(g_mutiftpt_record == padding, dim=-1, keepdim=True).squeeze(-1)
                # 利用attention得到prompt之间的关系
                g_mutiftpt_output, g_mutiftpt_attn_weights =  self.attention_layer(g_mutiftpt_record, g_mutiftpt_record, g_mutiftpt_record, key_padding_mask = key_padding_mask)
                # 对每个节点attention后所有的prompt求avg得到每个节点的最终混合prompt
                # g_mutiftpt_final_output = torch.mean(g_mutiftpt_output, dim=1) # 求平均，不好，因为有一些padding的embedding
                g_mutiftpt_final_output = g_mutiftpt_output[:,0,:] # BERT的方法，利用添加的readout_token的embedding
            else:    
                # 用求平均 如果只用'other_pt'就完全复刻GPF
                padding = torch.zeros(self.in_channels).to(device)
                # 找到每个节点prompt record中不为padding的行
                wo_padding_mask = torch.all(g_mutiftpt_record != padding, dim=-1, keepdim=True)
                # 统计每个节点不为padding的prompt数量
                node_prompt_len = wo_padding_mask.sum(1)
                g_mutiftpt_final_output = torch.where(node_prompt_len != 0, g_mutiftpt_record.sum(1) / node_prompt_len, padding)
            ######################################################################################
            



            # *****************************************  Prompt Pruned PART  ***************************************** #
            # # ######################################################################################
            # # 放在前面: 先修剪图再对特征进行多提示添加
            # # Prune edge index
            # edge_index = g.edge_index
            # cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            # # Define threshold t
            # threshold = 0.05
            # # Identify edges to keep
            # keep_edges = cosine_sim >= threshold
            # # Filter edge_index to only keep edges above the threshold
            # pruned_edge_index    = edge_index[:, keep_edges]
            # pruned_g_before_pt   = Data(x=g.x, edge_index=pruned_edge_index, y=g.y)
            # pruned_g_before_pt.x = self.add_pt(pruned_g_before_pt.x, g_mutiftpt_final_output)
            # graph_mutiftpt.append(pruned_g_before_pt)
            # # #####################################################################################

            # # ######################################################################################
            # # # 前后都不处理，直接加提示
            # g.x = self.add_pt(g.x, g_mutiftpt_final_output)
            # graph_mutiftpt.append(g)
            # # ######################################################################################

            ######################################################################################
            # 放在后面： 对图的特征进行多提示添加后根据添加prompt的特征修剪图
            g.x = self.add_pt(g.x, g_mutiftpt_final_output)
            # Prune edge index
            edge_index = g.edge_index
            cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            # Define threshold t
            threshold = self.pt_threshold
            # Identify edges to keep
            keep_edges = cosine_sim >= threshold
            # Filter edge_index to only keep edges above the threshold
            pruned_edge_index = edge_index[:, keep_edges]
            pruned_g_after_pt= Data(x=g.x, edge_index=pruned_edge_index, y=g.y)
            graph_mutiftpt.append(pruned_g_after_pt)
            ######################################################################################
            # *****************************************  loss PART  ***************************************** #


            # 当前图处理结束，保存当前图的start_index, 并且更新whole_batch中下一个图的start_index
            node_use_each_pt_whole_batch['g_start_index'].append(whole_batch_start_index)
            whole_batch_start_index = whole_batch_start_index + g.num_nodes   # 注意！不要用+=，会出现无法计算梯度问题, 常见的 inplace operation  x+=y，x*=y 是 inplace operation ，可以改为 x=x+y 和 x=x*y
            
        graph_mutiftpt_batch = Batch.from_data_list(graph_mutiftpt)
        return graph_mutiftpt_batch, node_use_each_pt_whole_batch










    def Tune(self, train_loader, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):  
            train_batch = train_batch.to(device)
            prompted_graph, node_use_each_pt_whole_batch = self.forward(train_batch, device)
            node_emb, graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch,  prompt_type = 'RobustPrompt-I')
            pre = answering(graph_emb)


            # *********************************************  loss PART  ********************************************* #
            ######################################################################################
            # loss_mse 提示整体以同质性假设为导向
            loss_mse = F.mse_loss(node_emb[prompted_graph.edge_index[0]], node_emb[prompted_graph.edge_index[1]])
            # print("loss_mse : ", loss_mse)
            ######################################################################################
            # loss_pt 针对每一个prompt让筛选节点的平均embedding和未筛选节点的平均embedding相似
            loss_pt = 0.
            # 不包括'other_pt'
            for pt in [key for key in self.pt_keys if key != 'other_pt']:
                # 全局的kl实现
                if self.kl_global:
                    node_use_each_pt_whole_batch[pt] = node_use_each_pt_whole_batch[pt].long()
                    all_batch_nodes    = torch.arange(0, node_emb.shape[0]).to(device)
                    mask = ~torch.isin(all_batch_nodes, node_use_each_pt_whole_batch[pt])
                    node_use_no_pt    = all_batch_nodes[mask]
                    # 这两个判断很重要，要不loss都是nan 
                    # kl的前提是必须满足两个判断 一是存在没有加提示的节点embedding作为指导，二是同时加了提示的节点也一定要存在
                    # 后者虽然概率很小，但也有可能不存在任何一个符合当前提示判断条件的节点，所以只要有一个不满足都不行
                    if len(node_use_no_pt) == 0 or len(node_use_each_pt_whole_batch[pt]) == 0: # 全局角度，直接跳过这个pt了
                        continue
                    global_pt_mean    = torch.mean(node_emb[node_use_each_pt_whole_batch[pt]], dim = 0)    # [ 1, hid_dim ] 一个batch只有一个 全局的
                    global_no_pt_mean = torch.mean(node_emb[node_use_no_pt], dim = 0)                      # [ 1, hid_dim ] 一个batch只有一个 全局的
                    loss_pt_kl = torch.nn.KLDivLoss()(F.log_softmax(global_pt_mean / self.temperature), F.softmax(global_no_pt_mean / self.temperature)) 
                    loss_pt += loss_pt_kl
             
                else:
                # 局部的用batch中每个图的kl实现
                    global_pt_batch = []
                    global_no_pt_batch = []
                    for i in range(len(node_use_each_pt_whole_batch[pt])):
                        start = node_use_each_pt_whole_batch['g_start_index'][i]
                        end   = node_use_each_pt_whole_batch['g_start_index'][i+1] if i != len(node_use_each_pt_whole_batch[pt]) - 1 else node_emb.shape[0]
                        all_nodes_g    = torch.arange(start, end).to(device)
                        mask = ~torch.isin(all_nodes_g, torch.tensor(node_use_each_pt_whole_batch[pt][i]).to(device))
                        node_use_no_pt = all_nodes_g[mask]
                        # 两个判断，局部角度，前者表示当前图不存在没加当前pt的node，但其他图可能存在；后者表示当前图不存在任何一个符合当前提示判断条件的节点，虽然概率很小
                        if len(node_use_no_pt) == 0 or len(node_use_each_pt_whole_batch[pt][i]) == 0: 
                            continue # 此时还没有跳出当前pt，只是跳过当前pt下的当前图
                        local_pt_mean    = torch.mean(node_emb[node_use_each_pt_whole_batch[pt][i]], dim = 0) # mean之后维度torch.Size([256])
                        local_no_pt_mean = torch.mean(node_emb[node_use_no_pt], dim = 0)                      # mean之后维度torch.Size([256])
                        global_pt_batch.append(local_pt_mean)
                        global_no_pt_batch.append(local_no_pt_mean)
                    # 因为mean之后的维度torch.Size([256])，只有一维，这里用cat需要变成[1,256]再拼接，所以直接用stack更方便，增加了一个维度
                    # 这里要加一个判断如果当前pt下所有图都没有node_use_no_pt，即global_pt_batch = [], global_no_pt_batch = [] 没有添加任何mean embedding
                    assert len(global_pt_batch) == len(global_no_pt_batch)
                    if len(global_pt_batch) == 0 or len(global_no_pt_batch) == 0:
                        continue # 这里跳出了当前pt，整个batch都判断完了都没有，所以不用进行kl loss
                    global_pt_batch    = torch.stack(global_pt_batch)        # [ shot_num * num_class, hid_dim ] 每个图一个
                    global_no_pt_batch = torch.stack(global_no_pt_batch)     # [ shot_num * num_class, hid_dim ] 每个图一个
                    loss_pt_kl = torch.nn.KLDivLoss()(F.log_softmax(global_pt_batch / self.temperature), F.softmax(global_no_pt_batch / self.temperature)) 
                    loss_pt += loss_pt_kl
            # print("loss_pt : ", loss_pt)
            ######################################################################################
            # loss_constraint 针对不同的pt进行约束      提供两种方法 norm计算矩阵的二范数，矩阵中所有元素平方求和后开根号
            muti_prompt, overlap_matrix = self.get_muti_prompt(node_use_each_pt_whole_batch, device)
            # 这是一个用'sim_pt': 0.6, 'degree_pt': 3, 'other_pt' : 'all'利用cosine_constraint得到的overlap_matrix
            #     tensor([[1.0000, 0.4804, 0.0000],                     
            #             [0.4804, 1.0000, 0.0000],                     
            #             [0.0000, 0.0000, 1.0000]], device='cuda:0')
            # 可以看到other和其他的pt根据节点重合度都为0，但sim_pt和degree_pt有交集
            if muti_prompt.shape[0] >= 2:
                # 方法一： 用cos相似度
                if self.cosine_constraint:
                    dot_product = torch.matmul(muti_prompt, muti_prompt.T)
                    norms = torch.norm(muti_prompt, dim=1)
                    muti_prompt_matrix = dot_product / (norms[:, None] * norms[None, :])
                    loss_constraint = torch.norm(muti_prompt_matrix - overlap_matrix)
                # 方法二： 和GPPT一样使用dot
                else:
                    loss_constraint = torch.norm(torch.mm(muti_prompt, muti_prompt.T) - overlap_matrix)
            else:
                loss_constraint = 0.
            # print("loss_constraint : ", loss_constraint)
            ######################################################################################
            # *********************************************  loss PART  ********************************************* #


            train_loss = lossfn(pre, train_batch.y) + self.weight_mse * loss_mse + self.weight_kl * loss_pt + self.weight_constraint * loss_constraint
            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
        return running_loss / len(train_loader)