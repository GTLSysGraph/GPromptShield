import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree

class RobustPrompt_I_Feat(torch.nn.Module):
    def __init__(self, in_channels: int, num_heads, muti_defense_pt_list):
        super(RobustPrompt_I_Feat, self).__init__()

        self.in_channels           = in_channels
        self.muti_defense_pt_list  = muti_defense_pt_list

        if 'sim_pt' in self.muti_defense_pt_list:
            self.prompt_sim_pt              = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        if 'degree_pt' in self.muti_defense_pt_list:
            self.prompt_degree_pt           = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 
        if 'out_detect_pt' in self.muti_defense_pt_list:
            self.prompt_out_detect_pt       = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 
        if 'other_pt' in self.muti_defense_pt_list:
            self.prompt_other_pt            = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 


        # attention  注意要用batch_first 因为nlp默认的是batch_first=False (L,N,Eq) L为目标序列长度，N为批量大小，Eq为查询嵌入维数embed_dim，batch_first=True时(N,L,Eq)
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim = self.in_channels * num_heads, num_heads = num_heads, dropout = 0.0, batch_first=True)
        self.readout_token   = torch.nn.Parameter(torch.randn(1, 1, self.in_channels))
        self.reset_parameters()


    def reset_parameters(self):
        if 'sim_pt' in self.muti_defense_pt_list:
            glorot(self.prompt_sim_pt)
        if 'degree_pt' in self.muti_defense_pt_list:
            glorot(self.prompt_degree_pt)
        if 'out_detect_pt' in self.muti_defense_pt_list:
            glorot(self.prompt_out_detect_pt) 
        if 'other_pt' in self.muti_defense_pt_list:
            glorot(self.prompt_other_pt)

        glorot(self.readout_token)


    def add_pt(self, x: torch.Tensor, specified_prompt):
        return x + specified_prompt  


    def get_muti_prompt(self, node_use_each_pt_whole_batch, device):
        muti_prompt   = []
        overlap_list  = []
        for name, param in self.named_parameters():
            if name.startswith('prompt'):         
                muti_prompt.append(param)
                # 这里是判断是 tensor还是 list 因为在上面分成了全局tensor和局部list
                # 明天实现 睡觉！
                overlap_list.append(node_use_each_pt_whole_batch[name.split('_', 1)[1]]) # example prompt[0] _ sim_pt[1] -> node_use_each_pt_whole_batch['sim_pt']

        muti_prompt = torch.stack(muti_prompt).squeeze()
        print(muti_prompt.shape)

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
        for pt in self.muti_defense_pt_list:
            # node_use_each_pt_whole_batch[pt] = torch.tensor([]).to(device)     # 用tensor直接把所有batch的都保存
            node_use_each_pt_whole_batch[pt] = []                            # 用list可以分开存每一个图的
        node_use_each_pt_whole_batch['g_start_index'] = []


        graph_mutiftpt = []
        for g in Batch.to_data_list(graph_batch):
            # 首先用-1初始化并拼接所有的pt长度
            g_mutiftpt_record = torch.ones(g.num_nodes, len(self.muti_defense_pt_list) * self.in_channels) * -1
            g_mutiftpt_record = g_mutiftpt_record.to(device)
            
            # 记录拼接所有pt的整体向量中每个pt的位置
            pt_range_dict = {}
            range_start = 0
            for pt in self.muti_defense_pt_list:
                pt_range_dict[pt] = [range_start, range_start + self.in_channels]
                range_start += self.in_channels
            # {'sim_pt': [0, 1433], 'degree_pt': [1433, 2866], 'other_pt': [2866, 4299]}

            x = g.x
            edge_index = g.edge_index
            # 用于记录当前图中所有defense pt用到的节点
            node_use_pt = torch.tensor([]).to(device)


            if 'sim_pt' in self.muti_defense_pt_list:
                # 相似度(cos (θ))值范围从-1(不相似)到+1(非常相似) nan代表孤立节点，对结果不造成影响
                x_norm = x / torch.sqrt(torch.sum(x * x,dim=1)).unsqueeze(-1)
                e = torch.sum(x_norm[edge_index[0]] * x_norm[edge_index[1]], dim = 1).unsqueeze(-1)
                row, col = edge_index
                c = torch.zeros(x_norm.shape[0], 1).to(device)
                c = c.scatter_add_(dim=0, index=col.unsqueeze(1), src=e)
                deg = degree(col, x.size(0), dtype=x.dtype).unsqueeze(-1) 
                csim = c /deg
                csim = csim.squeeze()
                node_use_sim_pt = torch.nonzero(csim <= 0.2).squeeze()
                # print(node_use_sim_pt)


                # 记录当前图中用到sim pt的node 局部的角度，为了后面筛选出一个图中没有用到任何defense pt的node
                node_use_pt = torch.concat((node_use_pt, node_use_sim_pt))
                # 将prompt放到当前图中用sim pt的指定节点上
                g_mutiftpt_record[node_use_sim_pt, pt_range_dict['sim_pt'][0] : pt_range_dict['sim_pt'][1]] = self.prompt_sim_pt
                # 记录每个图用到sim pt的node   全局的角度  注意这里一定要放在对当前图处理完后面，要不会改变节点索引
                node_use_sim_pt = node_use_sim_pt + whole_batch_start_index # 从当前图的start index进行记录  注意！不要用+=，会出现无法计算梯度问题
                # node_use_each_pt_whole_batch['sim_pt'] = torch.concat((node_use_each_pt_whole_batch['sim_pt'], node_use_sim_pt))
                node_use_each_pt_whole_batch['sim_pt'].append(node_use_sim_pt.tolist())
                # print(node_use_sim_pt)




            if 'degree_pt' in self.muti_defense_pt_list:
                deg = degree(col, x.size(0), dtype=x.dtype)
                node_use_degree_pt = torch.nonzero(deg <= 3).squeeze()


                # 记录当前图中用到degree pt的node 局部的角度，为了后面筛选出一个图中没有用到任何defense pt的node
                node_use_pt = torch.concat((node_use_pt, node_use_degree_pt))
                # 将prompt放到当前图中用degree pt的指定节点上
                g_mutiftpt_record[node_use_degree_pt, pt_range_dict['degree_pt'][0] : pt_range_dict['degree_pt'][1]] = self.prompt_degree_pt
                # 记录每个图用到degree pt的node 全局的角度  注意这里一定要放在对当前图处理完后面，要不会改变节点索引
                node_use_degree_pt = node_use_degree_pt + whole_batch_start_index # 从当前图的start index进行记录  注意！不要用+=，会出现无法计算梯度问题
                # node_use_each_pt_whole_batch['degree_pt'] = torch.concat((node_use_each_pt_whole_batch['degree_pt'],  node_use_degree_pt))
                node_use_each_pt_whole_batch['degree_pt'].append(node_use_degree_pt.tolist())
                # print(node_use_degree_pt)


            if 'out_detect_pt' in self.muti_defense_pt_list:
                pass




            if 'other_pt' in self.muti_defense_pt_list:
                # print('use other_pt, tips: Apply to the remaining nodes without adding pt! IF only other_pt,equal GPF')
                all_nodes    = torch.arange(0, g.num_nodes).to(device)
                all_pt_nodes = torch.unique(node_use_pt)
                mask = ~torch.isin(all_nodes, all_pt_nodes)
                node_use_no_pt = all_nodes[mask]
                # print(node_use_no_pt)


                # 将prompt放到当前图中没有用到任何defense pt的节点上
                g_mutiftpt_record[node_use_no_pt, pt_range_dict['other_pt'][0] : pt_range_dict['other_pt'][1]] = self.prompt_other_pt
                # 记录每个图没有用到任何defense pt的node 全局的角度 注意这里一定要放在对当前图处理完后面，要不会改变节点索引
                node_use_no_pt = node_use_no_pt + whole_batch_start_index # 从当前图的start index进行记录  注意！不要用+=，会出现无法计算梯度问题
                # node_use_each_pt_whole_batch['other_pt'] = torch.concat((node_use_each_pt_whole_batch['other_pt'], node_use_no_pt))
                node_use_each_pt_whole_batch['other_pt'].append(node_use_no_pt.tolist())
                # print(node_use_no_pt)




            # 加一个readout_token
            g_mutiftpt_record = g_mutiftpt_record.reshape(g.num_nodes, len(self.muti_defense_pt_list), self.in_channels)
            g_mutiftpt_record = torch.cat([self.readout_token.expand(g.num_nodes, 1, self.in_channels), g_mutiftpt_record], dim=1)

            # padding
            padding = torch.ones(self.in_channels) * -1
            padding = padding.to(device)
            key_padding_mask = torch.all(g_mutiftpt_record == padding, dim=-1, keepdim=True).squeeze(-1)

            # 利用attention得到prompt之间的关系
            g_mutiftpt_output, g_mutiftpt_attn_weights =  self.attention_layer(g_mutiftpt_record, g_mutiftpt_record, g_mutiftpt_record, key_padding_mask = key_padding_mask)

            # 对每个节点attention后所有的prompt求avg得到每个节点的最终混合prompt
            # g_mutiftpt_final_output = torch.mean(g_mutiftpt_output, dim=1) # 求平均，不好，因为有一些padding的embedding
            g_mutiftpt_final_output = g_mutiftpt_output[:,0,:] # BERT的方法，利用添加的readout_token的embedding
            g.x = self.add_pt(g.x, g_mutiftpt_final_output)
            graph_mutiftpt.append(g)


            # 当前图处理结束，保存当前图的start_index, 并且更新whole_batch中下一个图的start_index
            node_use_each_pt_whole_batch['g_start_index'].append(whole_batch_start_index)
            whole_batch_start_index = whole_batch_start_index + g.num_nodes   # 注意！不要用+=，会出现无法计算梯度问题, 常见的 inplace operation  x+=y，x*=y 是 inplace operation ，可以改为 x=x+y 和 x=x*y
            
            # print('num nodes in g: ', g.num_nodes)
            # print('whole_batch_start_index: ', whole_batch_start_index)


        graph_mutiftpt_batch = Batch.from_data_list(graph_mutiftpt)
        return graph_mutiftpt_batch, node_use_each_pt_whole_batch



    def Tune(self, train_loader, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        temperature = 1.0
        alpha = 0.2
        beta  = 0.2
        gamma = 0.2
        for batch_id, train_batch in enumerate(train_loader):  
            train_batch = train_batch.to(device)
            prompted_graph, node_use_each_pt_whole_batch = self.forward(train_batch, device)
            node_emb, graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch,  prompt_type = 'RobustPrompt_I')
            pre = answering(graph_emb)

            # muti_prompt, overlap_matrix = self.get_muti_prompt(node_use_each_pt_whole_batch, device)


            # loss_mse 提示整体以同质性假设为导向
            loss_mse = F.mse_loss(node_emb[prompted_graph.edge_index[0]], node_emb[prompted_graph.edge_index[1]])



            # loss_pt 针对每一个prompt让筛选节点的平均embedding和未筛选节点的平均embedding相似
            loss_pt = 0.
            for pt in self.muti_defense_pt_list:
                # 全局的kl实现
                # node_use_each_pt_whole_batch[pt] = node_use_each_pt_whole_batch[pt].long()
                # all_batch_nodes    = torch.arange(0, node_emb.shape[0]).to(device)
                # mask = ~torch.isin(all_batch_nodes, node_use_each_pt_whole_batch[pt])
                # node_use_no_pt = all_batch_nodes[mask]
                # global_pt_mean    = torch.mean(node_emb[node_use_each_pt_whole_batch[pt]], dim = 0)    # [ 1, hid_dim ] 一个batch只有一个 全局的
                # global_no_pt_mean = torch.mean(node_emb[node_use_no_pt], dim = 0)                      # [ 1, hid_dim ] 一个batch只有一个 全局的
                # loss_pt_kl = torch.nn.KLDivLoss(reduction= 'batchmean')(F.log_softmax(global_pt_mean / temperature), F.softmax(global_no_pt_mean / temperature)) 
                # loss_pt += loss_pt_kl


                # 局部的用batch中每个图的kl实现
                global_pt_batch = []
                global_no_pt_batch = []
                for i in range(len(node_use_each_pt_whole_batch[pt])):
                    start = node_use_each_pt_whole_batch['g_start_index'][i]
                    end   = node_use_each_pt_whole_batch['g_start_index'][i+1] if i != len(node_use_each_pt_whole_batch[pt]) - 1 else node_emb.shape[0]
                    all_nodes_g    = torch.arange(start, end).to(device)
                    mask = ~torch.isin(all_nodes_g, torch.tensor(node_use_each_pt_whole_batch[pt][i]).to(device))
                    node_use_no_pt = all_nodes_g[mask]
                    local_pt_mean    = torch.mean(node_emb[node_use_each_pt_whole_batch[pt][i]], dim = 0)
                    local_no_pt_mean = torch.mean(node_emb[node_use_no_pt], dim = 0)
                    global_pt_batch.append(local_pt_mean)
                    global_no_pt_batch.append(local_no_pt_mean)
                global_pt_batch    = torch.stack(global_pt_batch)        # [ shot_num * num_class, hid_dim ] 每个图一个
                global_no_pt_batch = torch.stack(global_no_pt_batch)     # [ shot_num * num_class, hid_dim ] 每个图一个
                loss_pt_kl = torch.nn.KLDivLoss(reduction= 'batchmean')(F.log_softmax(global_pt_batch / temperature), F.softmax(global_no_pt_batch / temperature)) 
                loss_pt += loss_pt_kl




            # loss_constraint 针对不同的pt进行约束      提供两种方法 norm计算矩阵的二范数，矩阵中所有元素平方求和后开根号
            # 方法一： 用cos相似度
            # dot_product = torch.matmul(muti_prompt, muti_prompt.T)
            # norms = torch.norm(muti_prompt, dim=1)
            # muti_prompt_matrix = dot_product / (norms[:, None] * norms[None, :])
            # loss_constraint = torch.norm(muti_prompt_matrix - overlap_matrix)

            # # 方法二： 和GPPT一样使用dot
            # loss_constraint = torch.norm(torch.mm(muti_prompt, muti_prompt.T) - overlap_matrix)




            train_loss = lossfn(pre, train_batch.y) + alpha * loss_mse + beta * loss_pt #+ gamma * loss_constraint
            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
        return running_loss / len(train_loader)