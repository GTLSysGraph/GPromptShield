import torchmetrics
import torch
from torch_geometric.data import Batch, Data
import torch.nn.functional as F

def RobustPromptInductiveEva(loader, tag, prompt, gnn, answering, num_class, device):
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
            batch = batch.to(device) 
 
            # idea 1
            # prompted_graph, num_nodes_induced_graphs, num_nodes_prompt_graphs = prompt(batch, pseudo_model)
            # idea 2
            # prompted_graph = prompt.add_robust_prompt(batch)
            # idear 3
            prompted_graph, _ = prompt(batch, tag, device)
            # raw
            # prompted_graph = batch

            ################
            pruned_prompt_batch_list = []
            for g_pt in Batch.to_data_list(prompted_graph):
                # Prune edge index
                edge_index = g_pt.edge_index
                cosine_sim = F.cosine_similarity(g_pt.x[edge_index[0]], g_pt.x[edge_index[1]])
                # Define threshold t
                threshold = 0.2
                # Identify edges to keep
                keep_edges = cosine_sim >= threshold
                # Filter edge_index to only keep edges above the threshold
                pruned_edge_index = edge_index[:, keep_edges]
                pruned_prompt_g  = Data(x=g_pt.x, edge_index=pruned_edge_index, y=g_pt.y)
                pruned_prompt_batch_list.append(pruned_prompt_g)
            pruned_prompt_graph = Batch.from_data_list(pruned_prompt_batch_list)
            ################


            node_emb, graph_emb = gnn(pruned_prompt_graph.x, pruned_prompt_graph.edge_index, pruned_prompt_graph.batch,  prompt_type = 'RobustPrompt_I')
            pre = answering(graph_emb)
            pred = pre.argmax(dim=1)  
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            print("{} Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(tag, batch_id, acc.item(), ma_f1.item()))

        acc = accuracy.compute()
        ma_f1 = macro_f1.compute()
        # print("Final True Acc: {:.4f} | Macro-F1: {:.4f}".format(acc.item(), ma_f1.item()))
        # roc = auroc.compute()
        # prc = auprc.compute()
        # return acc.item(), ma_f1.item(), roc.item(), prc.item()

        return acc.item(), ma_f1.item()


