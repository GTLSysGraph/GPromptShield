import torchmetrics
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data

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
    # 前后都不处理，直接加提示
    g_mutiftpt, _ = prompt.add_muti_pt(data, device)
    out = gnn(g_mutiftpt.x, g_mutiftpt.edge_index)
    ######################################################################################
    
    # ######################################################################################
    # # 放在后面： 对图的特征进行多提示添加后根据添加prompt的特征修剪图
    # g_mutiftpt, _ = prompt.add_muti_pt(data, device)
    # # 根据添加后的提示对图进行修剪
    # # Prune edge index
    # edge_index = g_mutiftpt.edge_index
    # cosine_sim = F.cosine_similarity(g_mutiftpt.x[edge_index[0]], g_mutiftpt.x[edge_index[1]])
    # # Define threshold t
    # threshold = 0.2
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


