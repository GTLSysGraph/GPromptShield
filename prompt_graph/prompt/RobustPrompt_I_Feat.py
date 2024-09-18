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
            self.low_sim_adjust_prompt    = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        if 'degree_pt' in self.muti_defense_pt_list:
            self.low_degree_adjust_prompt = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 
        if 'out_detect_pt' in self.muti_defense_pt_list:
            self.out_detect_adjust_prompt        = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 
        if 'other_pt' in self.muti_defense_pt_list:
            self.other_prompt             = torch.nn.Parameter(torch.Tensor(1, self.in_channels)) 


        # attention  注意要用batch_first 因为nlp默认的是batch_first=False (L,N,Eq) L为目标序列长度，N为批量大小，Eq为查询嵌入维数embed_dim，batch_first=True时(N,L,Eq)
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim = self.in_channels * num_heads, num_heads = num_heads, dropout = 0.0, batch_first=True)
        self.reset_parameters()


    def reset_parameters(self):
        if 'sim_pt' in self.muti_defense_pt_list:
            glorot(self.low_sim_adjust_prompt)
        if 'degree_pt' in self.muti_defense_pt_list:
            glorot(self.low_degree_adjust_prompt)
        if 'out_detect_pt' in self.muti_defense_pt_list:
            glorot(self.out_detect_adjust_prompt) 
        if 'other_pt' in self.muti_defense_pt_list:
            glorot(self.other_prompt)


        
        # for name, param in self.attention_layer.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_uniform_(param)  # Glorot 均匀分布初始化权重
        #     elif 'bias' in name:
        #         torch.nn.init.zeros_(param)  # 将偏置初始化为0



    def add(self, x: torch.Tensor, specified_prompt):
        return x + specified_prompt  


    def forward(self, graph_batch: Batch, device):
        graph_mutiftpt = []
        for g in Batch.to_data_list(graph_batch):
            # 首先用-1初始化并拼接所有的pt长度
            g_mutiftpt_record = torch.ones(g.num_nodes, len(self.muti_defense_pt_list) * self.in_channels) * -1
            g_mutiftpt_record = g_mutiftpt_record.to(device)
            pt_range_dict = {}
            range_start = 0
            for pt in self.muti_defense_pt_list:
                pt_range_dict[pt] = [range_start, range_start + self.in_channels]
                range_start += self.in_channels
            # {'sim_pt': [0, 1433], 'degree_pt': [1433, 2866], 'other_pt': [2866, 4299]}
      
            x = g.x
            edge_index = g.edge_index

            # 用于记录所有defense pt用到的节点
            node_use_pt = torch.tensor([]).to(device)

            if 'sim_pt' in self.muti_defense_pt_list:
                # print('use sim_pt')
                # 相似度(cos (θ))值范围从-1(不相似)到+1(非常相似) nan代表孤立节点，对结果不造成影响
                x_norm = x / torch.sqrt(torch.sum(x * x,dim=1)).unsqueeze(-1)
                e = torch.sum(x_norm[edge_index[0]] * x_norm[edge_index[1]], dim = 1).unsqueeze(-1)
                row, col = edge_index
                c = torch.zeros(x_norm.shape[0], 1).to(device)
                c = c.scatter_add_(dim=0, index=col.unsqueeze(1), src=e)
                # 获取节点的度
                deg = degree(col, x.size(0), dtype=x.dtype).unsqueeze(-1) 
                csim = c /deg
                csim = csim.squeeze()
                node_use_sim_pt = torch.nonzero(csim <= 0.6).squeeze()
                print(node_use_sim_pt)
                # 记录用sim pt到的node
                node_use_pt = torch.concat((node_use_pt, node_use_sim_pt))
                # 将prompt放到指定节点的指定位置上
                g_mutiftpt_record[node_use_sim_pt, pt_range_dict['sim_pt'][0] : pt_range_dict['sim_pt'][1]] = self.low_sim_adjust_prompt
                

            if 'degree_pt' in self.muti_defense_pt_list:
                # print('use degree_pt')
                deg = degree(col, x.size(0), dtype=x.dtype)
                node_use_degree_pt = torch.nonzero(deg <= 2).squeeze()
                print(node_use_degree_pt)
                # 记录用degree pt到的node
                node_use_pt = torch.concat((node_use_pt, node_use_degree_pt))
                # 将prompt放到指定节点的指定位置上
                g_mutiftpt_record[node_use_degree_pt, pt_range_dict['degree_pt'][0] : pt_range_dict['degree_pt'][1]] = self.low_degree_adjust_prompt

            if 'out_detect_pt' in self.muti_defense_pt_list:
                pass

            if 'other_pt' in self.muti_defense_pt_list:
                # print('use other_pt, tips: Apply to the remaining nodes without adding pt! IF only other_pt,equal GPF')
                all_nodes    = torch.arange(0, g.num_nodes).to(device)
                all_pt_nodes = torch.unique(node_use_pt)
                mask = ~torch.isin(all_nodes, all_pt_nodes)
                node_use_no_pt = all_nodes[mask]
                print(node_use_no_pt)
                g_mutiftpt_record[node_use_no_pt, pt_range_dict['other_pt'][0] : pt_range_dict['other_pt'][1]] = self.other_prompt




            g_mutiftpt_record = g_mutiftpt_record.reshape(g.num_nodes, len(self.muti_defense_pt_list), self.in_channels)
            padding = torch.ones(self.in_channels) * -1
            padding = padding.to(device)
            key_padding_mask = torch.all(g_mutiftpt_record == padding, dim=-1, keepdim=True).squeeze(-1)
            # print(key_padding_mask)

            test_node = 1
            print(g_mutiftpt_record[test_node])
            print(key_padding_mask[test_node])
            # 利用attention得到prompt之间的关系
            g_mutiftpt_output, g_mutiftpt_attn_weights =  self.attention_layer(g_mutiftpt_record, g_mutiftpt_record, g_mutiftpt_record, key_padding_mask = key_padding_mask)
            print(g_mutiftpt_output[test_node])
            print(g_mutiftpt_attn_weights[test_node])
            quit()

            # 对每个节点attention后所有的prompt求avg得到每个节点的最终混合prompt
            g_mutiftpt_final_output = torch.mean(g_mutiftpt_output, dim=1)

            # g.x[node_use_prompt] = g.x[node_use_prompt].add(self.low_sim_adjust_prompt)
            print(g_mutiftpt_final_output)
            quit()
            g.x = self.add(g.x, g_mutiftpt_final_output)
            print(g.x)
            quit()


        graph_mutiftpt.append(g)
        graph_mutiftpt_batch = Batch.from_data_list(graph_mutiftpt)
        return  graph_mutiftpt_batch


    def Tune(self, train_loader, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):  
            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch,device)
            node_emb, graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch,  prompt_type = 'RobustPrompt_I')
            pre = answering(graph_emb)

            # 提示以同质性假设为导向
            loss_mse = F.mse_loss(node_emb[prompted_graph.edge_index[0]], node_emb[prompted_graph.edge_index[1]])
            # 针对每一个prompt都让这些

            train_loss = lossfn(pre, train_batch.y) + loss_mse
            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
        return running_loss / len(train_loader)