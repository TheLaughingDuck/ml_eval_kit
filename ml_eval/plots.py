
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

def create_conf_matrix_fig(conf_matrix, classes, save_fig_as=None, title="", subtitle=None):
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


m = get_conf_matrix([0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 0, 2], n_classes=5)
create_conf_matrix_fig(m, save_fig_as=None, classes=["A", "B", "C", "D", "E"], title="Test Confusion Matrix", subtitle="This is a test subtitle")

# %%
