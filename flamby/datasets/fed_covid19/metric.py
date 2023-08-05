import numpy as np
from sklearn.metrics import f1_score

def metric(y_true, y_pred):
    y_pred = np.argmax(y_pred,axis=1).flatten()
    # The try except is needed because when the metric is batched some batches \
    # have one class only
    try:
        return f1_score(y_true, y_pred)
    except ValueError:
        return np.nan