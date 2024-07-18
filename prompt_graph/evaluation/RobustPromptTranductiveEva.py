import torchmetrics
import torch
from tqdm import tqdm

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

    prompted_x = prompt.add(data.x)
    out = gnn(prompted_x, data.edge_index)
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


