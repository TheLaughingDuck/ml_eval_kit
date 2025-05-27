from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import logging
logging.basicConfig(filename="logs/run.log",
                    level=logging.INFO,
                    format='%(asctime)s  %(levelname)s: %(message)s')
logging.info("Metrics module loaded successfully.")

#%%
def calculate_metrics(labels, predictions) -> dict:
    '''
    Function that calculates accuracy, precision, recall, and F1 score.

    Args:
        labels (list): List of true labels, given as consecutive, non-negative integers, i.e. 0,1,2,3,..., C.
        predictions (list): List of predicted labels, given as consecutive, non-negative integers, i.e. 0,1,2,3,..., C.
    '''

    
    n_classes = len(set(labels))
    if n_classes < 2:
        raise ValueError("At least two unique labels are required to calculate metrics.")

    n_samples = len(labels)
    if n_samples != len(predictions):
        raise ValueError("The number of labels and predictions must match.")
    if n_samples == 0:
        raise ValueError("The number of labels and predictions must be greater than zero.")
    if n_classes != len(set(predictions)):
        raise ValueError("The number of unique labels in predictions must match the number of unique labels in labels.")

    out = {"global":{}, "local":{}}

    # Calculate global metrics
    out["global"]["accuracy"] = accuracy_score(labels, predictions)
    out["global"]["n_missclassifications"] = int(n_samples-accuracy_score(labels, predictions, normalize=False))
    
    balanced_acc = recall_score(labels, predictions, average=None)
    out["global"]["balanced_accuracy"] = sum(balanced_acc) / n_classes

    out["global"]["precision"] = precision_score(labels, predictions, average='macro', zero_division=0)
    out["global"]["recall"] = recall_score(labels, predictions, average='macro', zero_division=0)
    out["global"]["f1_score"] = f1_score(labels, predictions, average='macro', zero_division=0)

    # Calculate per-class metrics
    out["local"]["precision"] = [float(i) for i in precision_score(labels, predictions, average=None, zero_division=0)]
    out["local"]["recall"] = [float(i) for i in recall_score(labels, predictions, average=None, zero_division=0)]

    if n_classes == 2:
        out["TPR"] = precision_score(labels, predictions, pos_label=1, zero_division=0)
        out["FPR"] = precision_score(labels, predictions, pos_label=0, zero_division=0)


    return out



labels = [0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2]
predic = [0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2]

calculate_metrics(labels, predic)

# %%

# %%
