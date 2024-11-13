import torchmetrics
import torch
from torch_geometric.data import Batch,Data
import torch.nn.functional as F

def AllInOneEva(loader, prompt, gnn, answering, num_class, device):
        prompt.eval()
        answering.eval()
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)

        # auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
        # auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

        accuracy.reset()
        macro_f1.reset()

        # auroc.reset()
        # auprc.reset()

        for batch_id, batch in enumerate(loader): 

            # ################
            # pruned_batch_list = []
            # for g in Batch.to_data_list(batch):
            #       # Prune edge index
            #       edge_index = g.edge_index
            #       cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            #       # Define threshold t
            #       threshold = 0.1
            #       # Identify edges to keep
            #       keep_edges = cosine_sim >= threshold
            #       # Filter edge_index to only keep edges above the threshold
            #       pruned_edge_index = edge_index[:, keep_edges]
            #       pruned_g          = Data(x=g.x, edge_index=pruned_edge_index,y=g.y, relabel_central_index= g.relabel_central_index, raw_index = g.raw_index, pseudo_label= g.pseudo_label)
            #       pruned_batch_list.append(pruned_g)
            # train_batch = Batch.from_data_list(pruned_batch_list)
            # ################

            # ###############
            # pruned_batch_list = []
            # for g in Batch.to_data_list(batch):
            #     g = g.to(device)
            #     logits_ptb = gnn(g.x, g.edge_index)
            #     logits_ptb = torch.concat((logits_ptb, g.x), dim=1)
            #     features_edge = torch.concat((logits_ptb[g.edge_index[0]], logits_ptb[g.edge_index[1]]), dim=1)
            #     remove_flag = torch.zeros(g.edge_index.shape[1], dtype=torch.bool).to(device)
            #     for k in range(len(detectors)):
            #             output = F.sigmoid(detectors[k](features_edge)).squeeze(-1)
            #             remove_flag = torch.where(output > 0.1, True, remove_flag)
            #     keep_edges = remove_flag == False
            #     pruned_edge_index = g.edge_index[:, keep_edges]
            #     pruned_g          = Data(x=g.x, edge_index=pruned_edge_index,y=g.y, relabel_central_index= g.relabel_central_index, raw_index = g.raw_index, pseudo_label= g.pseudo_label)
            #     pruned_batch_list.append(pruned_g)
            # batch = Batch.from_data_list(pruned_batch_list)
            # ###############




            batch = batch.to(device) 
            prompted_graph = prompt(batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # print(graph_emb)
            pre = answering(graph_emb)
            pred = pre.argmax(dim=1)  
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            print("Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id, acc.item(), ma_f1.item()))

        acc = accuracy.compute()
        ma_f1 = macro_f1.compute()
        # print("Final True Acc: {:.4f} | Macro-F1: {:.4f}".format(acc.item(), ma_f1.item()))
        # roc = auroc.compute()
        # prc = auprc.compute()
        # return acc.item(), ma_f1.item(), roc.item(), prc.item()

        return acc.item(), ma_f1.item()




def AllInOneEvaWithoutAnswer(loader, prompt, gnn, num_class, device):
        prompt.eval()
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
        accuracy.reset()
        macro_f1.reset()
        for batch_id, test_batch in enumerate(loader):
            test_batch = test_batch.to(device)
            emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
            pg_batch = prompt.token_view()
            pg_batch = pg_batch.to(device)
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            pre = torch.softmax(dot, dim=1)

            y = test_batch.y
            pre_cla = torch.argmax(pre, dim=1)

            acc = accuracy(pre_cla, y)
            ma_f1 = macro_f1(pre_cla, y)

        acc = accuracy.compute()
        ma_f1 = macro_f1.compute()
        return acc





def AllInOneGraphEva(loader, prompt, gnn, answering, num_class, device):
    prompt.eval()
    answering.eval()

    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            batch = batch.to(device)
            prompted_graph = prompt(batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            pred = pre.argmax(dim=1)

            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(pre, batch.y)
            prc = auprc(pre, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    return acc.item(), ma_f1.item(), roc.item(), prc.item()