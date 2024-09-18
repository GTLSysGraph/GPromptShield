import torchmetrics
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

def GPFEva(loader, gnn, prompt, answering, num_class, device):
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
    with torch.no_grad(): 
        for batch_id, batch in enumerate(loader): 

            ################
            # pruned_batch_list = []
            # for g in Batch.to_data_list(batch):
            #     print(g)
            #     # Prune edge index
            #     edge_index = g.edge_index
            #     cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            #     # Define threshold t
            #     threshold = 0.2
            #     # Identify edges to keep
            #     keep_edges = cosine_sim >= threshold
            #     # Filter edge_index to only keep edges above the threshold
            #     pruned_edge_index = edge_index[:, keep_edges]
            #     pruned_g          = Data(x=g.x, edge_index=pruned_edge_index,y=g.y, relabel_central_index= g.relabel_central_index, raw_index = g.raw_index, pseudo_label= g.pseudo_label)
            #     print(pruned_g)
            #     quit()
            #     pruned_batch_list.append(pruned_g)
            # batch = Batch.from_data_list(pruned_batch_list)
            ################

            ################
            # pruned_batch_list = []
            # for g in Batch.to_data_list(batch):
            #     print(g)
            #     g = g.to(device)
            #     logits_ptb = gnn(g.x, g.edge_index)
            #     logits_ptb = torch.concat((logits_ptb, g.x), dim=1)
            #     features_edge = torch.concat((logits_ptb[g.edge_index[0]], logits_ptb[g.edge_index[1]]), dim=1)
            #     remove_flag = torch.zeros(g.edge_index.shape[1], dtype=torch.bool).to(device)
            #     for k in range(len(detectors)):
            #         output = F.sigmoid(detectors[k](features_edge)).squeeze(-1)
            #         remove_flag = torch.where(output > 0.4, True, remove_flag)
            #     keep_edges = remove_flag == False
            #     pruned_edge_index = g.edge_index[:, keep_edges]
            #     pruned_g          = Data(x=g.x, edge_index=pruned_edge_index,y=g.y, relabel_central_index= g.relabel_central_index, raw_index = g.raw_index, pseudo_label= g.pseudo_label)
            #     print(pruned_g)
            #     pruned_batch_list.append(pruned_g)
            # batch = Batch.from_data_list(pruned_batch_list)
            ################



            batch = batch.to(device) 
            batch.x = prompt.add(batch.x)
            out = gnn(batch.x, batch.edge_index, batch.batch)
            if answering:
                out = answering(out)  
            pred = out.argmax(dim=1)  

            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            # roc = auroc(out, batch.y)
            # prc = auprc(out, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item()))

                # print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

            # print(acc)
    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    # roc = auroc.compute()
    # prc = auprc.compute()
       
    return acc.item(), ma_f1.item()#, roc.item(),prc.item()




# def GPFEva(loader, gnn, prompt, answering, device):
#     prompt.eval()
#     if answering:
#         answering.eval()
#     correct = 0
#     for batch in loader: 
#         batch = batch.to(device) 
#         batch.x = prompt.add(batch.x)
#         out = gnn(batch.x, batch.edge_index, batch.batch)
#         if answering:
#             out = answering(out)  
#         pred = out.argmax(dim=1)  
#         correct += int((pred == batch.y).sum())  
#     acc = correct / len(loader.dataset)
#     return acc  