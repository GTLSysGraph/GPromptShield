from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import os
import pandas as pd
from torch_geometric.datasets import NELL,  Planetoid


def induced_graphs_from_edges(data, device, smallest_size=5, largest_size=20):
    induced_graph_list = []

    edge_index = data.edge_index

    for edge_id in range(edge_index.size(1)):
        src_node = edge_index[0, edge_id].item()
        tgt_node = edge_index[1, edge_id].item()
        current_label = 1  # Label is 1 if there's an edge between the nodes

        current_hop = 1

        subset, _, _, _ = k_hop_subgraph(node_idx=src_node, num_hops=current_hop,
                                        edge_index=edge_index, relabel_nodes=True)
        subset_tgt, _, _, _ = k_hop_subgraph(node_idx=tgt_node, num_hops=current_hop,
                                            edge_index=edge_index, relabel_nodes=True)
        subset = torch.unique(torch.cat([subset, subset_tgt]))

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset_src, _, _, _ = k_hop_subgraph(node_idx=src_node, num_hops=current_hop,
                                                edge_index=edge_index)
            subset_tgt, _, _, _ = k_hop_subgraph(node_idx=tgt_node, num_hops=current_hop,
                                                edge_index=edge_index)
            subset = torch.unique(torch.cat([subset_src, subset_tgt]))

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.arange(data.x.size(0))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), candidate_nodes])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size]
            subset = torch.unique(torch.cat([torch.LongTensor([src_node, tgt_node]).to(device), subset]))

        sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=torch.tensor([current_label], dtype=torch.long))
        induced_graph_list.append(induced_graph)
        if edge_id % 1000 == 0:
            print(edge_id)
        if edge_id >10000:
            break
    
    
    # Add non-connected pairs
    negative_sample_count = 0
    max_negative_samples = 10000

    for src_node in range(data.x.size(0)):
        for tgt_node in range(src_node + 1, data.x.size(0)):
            if not ((edge_index[0] == src_node) & (edge_index[1] == tgt_node)).any() and not ((edge_index[0] == tgt_node) & (edge_index[1] == src_node)).any():
                current_label = 0  # Label is 0 if there's no edge between the nodes
                subset = torch.LongTensor([src_node, tgt_node]).to(device)
                
                sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
                x = data.x[subset]

                induced_graph = Data(x=x, edge_index=sub_edge_index, y=torch.tensor([current_label], dtype=torch.long))
                induced_graph_list.append(induced_graph)

                negative_sample_count += 1
                if negative_sample_count%1000==0:
                    print(negative_sample_count)

                if negative_sample_count >= max_negative_samples:
                    break
        if negative_sample_count >= max_negative_samples:
            break
    

    return induced_graph_list
   

args = get_args()
seed_everything(args.seed)


print('dataset_name', args.dataset_name)

# args.dataset_name = 'NELL'
# dataset = NELL(root='./dataset/NELL')
# input_dim = 61278

args.dataset_name = 'Cora'
dataset = Planetoid(root='data/Planetoid', name='Cora')
input_dim = 1433

dataset = induced_graphs_from_edges(new_data, device, smallest_size=1, largest_size=30)
# args.prompt_type = 'All-in-one'
print('device:'+str(args.device))
tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = 1000,
                    shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = 1024, dataset = dataset, input_dim = input_dim, output_dim = 2)
pre_train_type = tasker.pre_train_type


_, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _= tasker.run()
  
print("Final Accuracy {:.2f}±{:.2f}(std)".format(test_acc*100, std_test_acc*100)) 
print("Final F1 {:.2f}±{:.2f}(std)".format(f1*100,std_f1*100)) 
print("Final AUROC {:.2f}±{:.2f}(std)".format(roc*100, std_roc*100)) 

print(" {:.2f}±{:.2f}|{:.2f}±{:.2f}|{:.2f}±{:.2f}".format(test_acc*100, std_test_acc*100, f1*100,std_f1*100, roc*100, std_roc*100)) 
