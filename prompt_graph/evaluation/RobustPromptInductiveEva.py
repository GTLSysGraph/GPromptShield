import torchmetrics
import torch
from torch_geometric.data import Batch, Data
import torch.nn.functional as F

def RobustPromptInductiveEva(loader, gnn, prompt, answering, num_class, device):
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
 

            # ######################################################################################
            # # 放在前面: 先修剪图再对特征进行多提示添加
            # pruned_batch_before_pt_list = []
            # for g in Batch.to_data_list(batch):
            #     # Prune edge index
            #     edge_index = g.edge_index
            #     cosine_sim = F.cosine_similarity(g.x[edge_index[0]], g.x[edge_index[1]])
            #     # Define threshold t
            #     threshold = 0.2
            #     # Identify edges to keep
            #     keep_edges = cosine_sim >= threshold
            #     # Filter edge_index to only keep edges above the threshold
            #     pruned_edge_index   = edge_index[:, keep_edges]
            #     pruned_g_before_pt  = Data(x=g_pt.x, edge_index=pruned_edge_index, y=g_pt.y)
            #     pruned_batch_before_pt_list.append(pruned_g_before_pt)
            # pruned_batch_before_pt = Batch.from_data_list(pruned_batch_before_pt_list)
            # prompted_graph, _      = prompt(pruned_batch_before_pt, tag, device)
            # node_emb, graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch,  prompt_type = 'RobustPrompt-I')
            # ######################################################################################


            ######################################################################################
            # 前后都不处理，直接加提示 这里需要注意下，inductive的修剪图是加在Tune里面的
            prompted_graph, _   = prompt(batch, device)
            node_emb, graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch,  prompt_type = 'RobustPrompt-I')
            ######################################################################################


            # ######################################################################################
            # # 放在后面： 对图的特征进行多提示添加后根据添加prompt的特征修剪图
            # prompted_graph, _ = prompt(batch, tag, device)
            # pruned_batch_after_pt_list = []
            # for g_pt in Batch.to_data_list(prompted_graph):
            #     # Prune edge index
            #     edge_index = g_pt.edge_index
            #     cosine_sim = F.cosine_similarity(g_pt.x[edge_index[0]], g_pt.x[edge_index[1]])
            #     # Define threshold t
            #     threshold = 0.5
            #     # Identify edges to keep
            #     keep_edges = cosine_sim >= threshold
            #     # Filter edge_index to only keep edges above the threshold
            #     pruned_edge_index = edge_index[:, keep_edges]
            #     pruned_g_after_pt  = Data(x=g_pt.x, edge_index=pruned_edge_index, y=g_pt.y)
            #     pruned_batch_after_pt_list.append(pruned_g_after_pt)
            # pruned_batch_after_pt = Batch.from_data_list(pruned_batch_after_pt_list)
            # node_emb, graph_emb = gnn(pruned_batch_after_pt.x, pruned_batch_after_pt.edge_index, pruned_batch_after_pt.batch,  prompt_type = 'RobustPrompt-I')
            # ######################################################################################



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







def RobustPromptInductiveGraphEva(loader, gnn, prompt, answering, num_class, device):
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
                ######################################################################################
                # 前后都不处理，直接加提示 这里需要注意下，inductive的修剪图是加在Tune里面的
                prompted_graph, _   = prompt(batch, device)
                node_emb, graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch,  prompt_type = 'RobustPrompt-I')
                ######################################################################################
                pre = answering(graph_emb)
                pred = pre.argmax(dim=1)  

                acc = accuracy(pred, batch.y)
                ma_f1 = macro_f1(pred, batch.y)
                roc = auroc(pre, batch.y)
                prc = auprc(pre, batch.y)
                if len(loader) > 2:
                    print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))
                    
        acc = accuracy.compute()
        ma_f1 = macro_f1.compute()
        roc = auroc.compute()
        prc = auprc.compute()

        return acc.item(), ma_f1.item(), roc.item(), prc.item()


