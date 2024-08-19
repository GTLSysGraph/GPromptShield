import torch 
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data

def prepare_structured_data(graph_data: Data):
    r"""Prepare structured <i,k,j> format link prediction data"""
    node_idx = torch.LongTensor([i for i in range(graph_data.num_nodes)])
    
    # 如果有自环，就不需要重复添加了，直接用
    from torch_geometric.utils import contains_self_loops
    if not contains_self_loops(graph_data.edge_index):
        self_loop = torch.stack([node_idx, node_idx], dim=0)
        edge_index = torch.cat([graph_data.edge_index, self_loop], dim=1)
    else:
        edge_index = graph_data.edge_index

    v, a, b = structured_negative_sampling(edge_index, graph_data.num_nodes)
    data = torch.stack([v, a, b], dim=1)
    # (num_edge, 3)
    #   for each entry (i,j,k) in data, (i,j) is a positive sample while (i,k) forms a negative sample
    return data