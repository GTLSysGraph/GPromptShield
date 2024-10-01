import torchmetrics
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data

from sklearn.metrics.pairwise import cosine_similarity
def get_reliable_neighbors(prompt_g, x, k, degree_threshold):
        num_nodes = prompt_g.num_nodes  # 节点数量

        adj = torch.zeros((num_nodes, num_nodes), device=prompt_g.edge_index.device)
        adj[prompt_g.edge_index[0], prompt_g.edge_index[1]] = 1  # 假设无向图，填充为1

        degree = adj.sum(dim=1)
        degree_mask = degree > degree_threshold
        assert degree_mask.sum().item() >= k
        sim = cosine_similarity(x.to('cpu'))
        sim = torch.FloatTensor(sim).to('cuda')
        sim[:, degree_mask == False] = 0
        _, top_k_indices = sim.topk(k=k, dim=1)
        for i in range(adj.shape[0]):
            adj[i][top_k_indices[i]] = 1
            adj[i][i] = 0

        edge_index = torch.stack(adj.nonzero(as_tuple=True),dim=0)
        prompt_g_power = Data(edge_index=edge_index, num_nodes=adj.size(0), x=x)
        return prompt_g_power



def RobustPromptTranductiveEva(data, mask, gnn, prompt, answering, num_class, device):
    prompt.eval()
    if answering:
        answering.eval()

    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    # auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    # auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    # auroc.reset()
    # auprc.reset()


    # ######################################################################################
    # # 放在前面: 先修剪图再对特征进行多提示添加
    # # Prune edge index
    # edge_index = data.edge_index
    # cosine_sim = F.cosine_similarity(data.x[edge_index[0]], data.x[edge_index[1]])
    # # Define threshold t
    # threshold = 0.05
    # # Identify edges to keep
    # keep_edges = cosine_sim >= threshold
    # # Filter edge_index to only keep edges above the threshold
    # pruned_edge_index = edge_index[:, keep_edges]
    # pruned_g  = Data(x=data.x, edge_index=pruned_edge_index, y=data.y)
    # g_mutiftpt, _ = prompt.add_muti_pt(pruned_g, device)
    # out = gnn(g_mutiftpt.x, g_mutiftpt.edge_index)
    # ######################################################################################

    ######################################################################################
    # 根据提示增强可靠邻居
    # g_mutiftpt, _    = prompt.add_muti_pt(data, device)
    # g_mutiftpt_power = get_reliable_neighbors(g_mutiftpt, g_mutiftpt.x.detach(), k=3, degree_threshold=6)
    # out = gnn(g_mutiftpt_power.x, g_mutiftpt_power.edge_index)
    ######################################################################################
    
    #####################################################################################
    # 前后都不处理，直接加提示
    g_mutiftpt, _    = prompt.add_muti_pt(data, device)
    out = gnn(g_mutiftpt.x, g_mutiftpt.edge_index)
    #####################################################################################


    # ######################################################################################
    # # 放在后面： 对图的特征进行多提示添加后根据添加prompt的特征修剪图
    # g_mutiftpt, _ = prompt.add_muti_pt(data, device)
    # # 根据添加后的提示对图进行修剪
    # # Prune edge index
    # edge_index = g_mutiftpt.edge_index
    # cosine_sim = F.cosine_similarity(g_mutiftpt.x[edge_index[0]], g_mutiftpt.x[edge_index[1]])
    # # Define threshold t
    # threshold = 0.1
    # # Identify edges to keep
    # keep_edges = cosine_sim >= threshold
    # # Filter edge_index to only keep edges above the threshold
    # pruned_edge_index = edge_index[:, keep_edges]
    # pruned_g_mutiftpt  = Data(x=g_mutiftpt.x, edge_index=pruned_edge_index, y=g_mutiftpt.y)
    # out = gnn(pruned_g_mutiftpt.x, pruned_g_mutiftpt.edge_index)
    # ######################################################################################






    
    if answering:
        out = answering(out)  
    pred = out.argmax(dim=1)  

    # roc = auroc(out, batch.y)
    # prc = auprc(out, batch.y)`

    acc = accuracy(pred[mask], data.y[mask])
    f1 = macro_f1(pred[mask], data.y[mask])
    # roc = auroc(out[mask], data.y[mask]) 
    # prc = auprc(out[mask], data.y[mask]) 
    return acc.item(), f1.item() #, roc.item(),prc.item()


