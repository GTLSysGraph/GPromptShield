import torchmetrics
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data

def GPFTranductiveEva(data, mask, gnn, prompt, answering, num_class, device):
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





    # ################
    # # 放在前面: 先修剪图再对特征添加prompt
    # # Prune edge index
    # edge_index = data.edge_index
    # cosine_sim = F.cosine_similarity(data.x[edge_index[0]], data.x[edge_index[1]])
    # # Define threshold t
    # threshold = 0.5
    # # Identify edges to keep
    # keep_edges = cosine_sim >= threshold
    # # Filter edge_index to only keep edges above the threshold
    # pruned_edge_index = edge_index[:, keep_edges]
    # pruned_g  = Data(x=data.x, edge_index=pruned_edge_index, y=data.y)
    # prompted_x = prompt.add(pruned_g.x)
    # out = gnn(prompted_x, pruned_g.edge_index)
    # ################

    ###############
    # 前后都不处理，直接加提示
    prompted_x = prompt.add(data.x)
    out = gnn(prompted_x, data.edge_index)
    ###############


    # ################
    # # 放在后面： 先加提示后根据添加prompt的特征修剪图
    # prompted_x = prompt.add(data.x)
    # # Prune edge index
    # edge_index = data.edge_index
    # cosine_sim = F.cosine_similarity(prompted_x[edge_index[0]], prompted_x[edge_index[1]])
    # # Define threshold t
    # threshold = 0.6
    # # Identify edges to keep
    # keep_edges = cosine_sim >= threshold
    # # Filter edge_index to only keep edges above the threshold
    # pruned_edge_index = edge_index[:, keep_edges]
    # pruned_g  = Data(x=prompted_x, edge_index=pruned_edge_index, y=data.y)
    # out = gnn(prompted_x, pruned_g.edge_index)
    # ################






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

