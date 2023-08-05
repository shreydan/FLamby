import numpy as np
from sklearn.metrics import f1_score

def metric(predictions,gt):
    predictions = np.argmax(predictions,axis=1).flatten()
    gt = gt.flatten()
    f1 = f1_score(y_true=gt,y_pred=predictions)
    return f1