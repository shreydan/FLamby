import torch
from torchmetrics.functional import f1_score

def metric(predictions,gt):
    preds = torch.argmax(predictions, dim=1)
    # auroc = roc_auc_score(y_true=gt.numpy().flatten(),y_score=preds.numpy().flatten())
    f1 = f1_score(preds=predictions.squeeze(),target=gt.squeeze(),task='binary')
#     cfm = confusion_matrix(y_true=gt.numpy().flatten(),y_pred=preds.flatten(),labels=list(range(2)))
    return f1