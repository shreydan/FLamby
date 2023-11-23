import numpy as np
from sklearn.metrics import cohen_kappa_score

def metric(y_true, y_pred):
    y_pred = np.argmax(y_pred,axis=1).flatten()
    return cohen_kappa_score(y1=y_true,y2=y_pred,labels=list(range(3)),weights='quadratic')