

def get_conf_matrix(all_preds, all_targets, n_classes=3) -> list:
    '''
    Takes two integer lists of all target classes, and all predictions by some classifier.

    Returns a confusion matrix, with true class on rows, and predicted class on the columns,
    as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    '''
    matrix = [[0 for i in range(n_classes)] for i in range(n_classes)]

    print(f"There are {len(all_preds)} predictions in all_preds.")

    for i in range(n_classes):
        for j in range(n_classes):
            for tar, pre in zip(all_targets, all_preds):
                if tar == i and pre == j:
                    matrix[i][j] += 1
    
    return(matrix)



import matplotlib.pyplot as plt
import numpy as np

def create_conf_matrix_fig(conf_matrix, save_fig_as=None, epoch=None, title="", n_classes=3, subtitle=None):
    '''
    Takes confusion matrices (on training and validation data),
    and creates a figure with them. Saves as a png.

    The true classes are on the rows, and the predicted values on the columns.
    '''
    fig, axs = plt.subplots(ncols=1)
    #fig.tight_layout(rect=(0,0,1,0.999))

    # axs[0].matshow(train_mat)
    # for (i, j), z in np.ndenumerate(train_mat):
    #     axs[0].text(j, i, '{}'.format(z), ha='center', va='center')
    # axs[0].set_yticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    # axs[0].set_xticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    # axs[0].set_xlabel("True value")
    # axs[0].set_ylabel("Prediction")
    # axs[0].set_title("Training data")

    if n_classes == 3:
        labels = ["Gli", "Epe", "Med"]
    elif n_classes == 2:
        labels = ["Supra", "Infra"]
    else:
        raise ValueError("Unsupported number of classes")

    axs.matshow(conf_matrix, alpha=0.3, cmap="Blues")
    for (i, j), z in np.ndenumerate(conf_matrix):
        axs.text(j, i, '{}'.format(z), ha='center', va='center', size="xx-large")
    axs.set_yticks(ticks=[i for i in range(n_classes)], labels=labels, fontsize=12)
    axs.set_xticks(ticks=[i for i in range(n_classes)], labels=labels, fontsize=12)
    axs.set_ylabel("True value", fontsize=16)
    axs.set_xlabel("Prediction", fontsize=16)
    #axs.set_title("Validation data")

    if subtitle != None: plt.title(subtitle)

    if save_fig_as != None:
        #start_date = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}", save_fig_as).group(0)
        #fig.suptitle(title+" (Epoch "+str(epoch)+")", fontsize=16)
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.savefig(save_fig_as)
    else:
        if title == "": title = "Confusion matrix"
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.show()
