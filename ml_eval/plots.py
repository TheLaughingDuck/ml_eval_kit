
#%%
def get_conf_matrix(all_preds, all_targets, n_classes=None, verbose=0) -> list:
    '''
    Takes two integer lists of all target classes, and all predictions by some classifier.

    Returns a confusion matrix, with true class on rows, and predicted class on the columns,
    as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Args:
        all_preds (list): List of predicted classes, as consecutive integers, i.e. 0,1,2,3,..., C.
        all_targets (list): List of true classes, as consecutive integers, i.e. 0,1,2,3,..., C.
        n_classes (int): Number of classes. If None, it is inferred from the data. Specify if some classes might be missing in the predictions.

        verbose (int): Verbosity level. If 0, no output is printed. If 1, prints the number of predictions in all_preds.
    '''
    # Infer number of classes from the data if not specified
    if n_classes is None: n_classes = max(max(all_preds), max(all_targets)) + 1

    # Setup empty confusion matrix
    matrix = [[0 for i in range(n_classes)] for i in range(n_classes)]

    if verbose > 0:
        print(f"Number of classes: {n_classes}")
        print(f"There are {len(all_targets)} targets in all_targets.")
        print(f"There are {len(all_preds)} predictions in all_preds.")

    for i in range(n_classes):
        for j in range(n_classes):
            for tar, pre in zip(all_targets, all_preds):
                if tar == i and pre == j:
                    matrix[i][j] += 1
    
    return(matrix)



import matplotlib.pyplot as plt
import numpy as np

def create_conf_matrix_fig(conf_matrix, classes=None, save_fig_as=None, title="", subtitle=None):
    '''
    Takes confusion matrices (on training and validation data),
    and creates a figure with them. Saves as a png.

    The true classes are on the rows, and the predicted values on the columns.

    Args:
        conf_matrix (list): Confusion matrix, as returned by get_conf_matrix().
        save_fig_as (str): Path to save the figure as a png file.
        title (str): Title of the figure.
        subtitle (str): Subtitle of the figure.
    '''
    fig, axs = plt.subplots(ncols=1)

    n_classes = len(conf_matrix)
    if classes is None: classes = [str(i) for i in range(n_classes)]

    axs.matshow(conf_matrix, alpha=0.3, cmap="Blues")
    for (i, j), z in np.ndenumerate(conf_matrix):
        axs.text(j, i, '{}'.format(z), ha='center', va='center', size="xx-large")
    axs.set_yticks(ticks=[i for i in range(n_classes)], labels=classes, fontsize=12)
    axs.set_xticks(ticks=[i for i in range(n_classes)], labels=classes, fontsize=12)
    axs.set_ylabel("True value", fontsize=16)
    axs.set_xlabel("Prediction", fontsize=16)
    #axs.set_title("Validation data")

    if subtitle != None: plt.title(subtitle)

    if save_fig_as != None:
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.savefig(save_fig_as)
    else:
        if title == "": title = "Confusion matrix"
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.show()

#%%
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pandas as pd

def create_roc_fig(labels, scores, save_fig_as=None, title="ROC Curves"):
    '''
    Creates a figure with ROC curves for all classes.
    '''
    #raise NotImplementedError("create_roc_fig() is not implemented yet.")

    # Load the true labels and predictions from the provided paths
    with open(labels, 'r') as f:
        labels = []
        for line in f.readlines():
            labels.append(int(line.strip().rstrip(",")))
    
    scores = pd.read_csv("C:/Users/jorst/Documents/Github/ml_eval_kit/some_dir/y_scores.csv", header=None).values.tolist()

    n_classes = len(set(labels))  # Number of unique classes
    colors = ["red", "blue", "green", "orange"]

    # Figure setup
    plt.figure()
    plt.tight_layout()
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.grid(True, linestyle='--')

    # Calculate ROC values
    for i in range(n_classes):
        roc_values = roc_curve(labels, [obs[i] for obs in scores], pos_label=1)
        plt.plot(roc_values[0], roc_values[1], c=colors[i], label=f"Class {i}")
    auc = roc_auc_score(labels, scores, multi_class='ovr', average='macro')
    
    # Draw the diagonal line
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    
    plt.suptitle(title, fontsize=22) # Main title
    plt.title(f"AUC: {round(auc, 2)} (unweighted average)") # Subtitle


    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend()

    # Save fig
    if save_fig_as is not None:
        plt.savefig(save_fig_as)
    else:
        plt.show()

# %%
#create_roc_fig("C:/Users/jorst/Documents/Github/ml_eval_kit/some_dir/y_true.csv", "C:/Users/jorst/Documents/Github/ml_eval_kit/some_dir/y_scores.csv")
# %%
