import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from matplotlib.font_manager import FontProperties


# This function calculates the confusion matrix and visualizes it
def plot_confusion_matrix(y_test, y_pred, path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    classes_pred = [str(i) for i in np.unique(y_pred)]
    classes_test = [str(i) for i in np.unique(y_test)]

    classes = None
    if len(classes_pred)>len(classes_test):
        classes = classes_pred
    else:
        classes = classes_test
    # In case the confusion matrix should be normalized
    if normalize:
        t = cm.sum(axis=1)[:, np.newaxis]
        for i in t:
            if i[0] == 0:
                i[0] = 1
        cm = cm.astype('float') / t

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.around(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)
    plt.close()

def plot_graphs(discriminator_losses, discriminator_accuracies, generator_losses, names, base_path, inception_scores=None):

    f = plt.figure()
    ax = plt.subplot(111)
    for i in range(len(discriminator_losses)):
        ax.plot(np.arange(len(discriminator_losses[i])), discriminator_losses[i], label="{}".format(names[i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.title("Training Discriminator Loss")
    plt.xlabel("Epochs")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(base_path+"d_loss.png")
    plt.close()

    f = plt.figure()
    ax = plt.subplot(111)
    for i in range(len(generator_losses)):
        ax.plot(np.arange(len(generator_losses[i])), generator_losses[i], label="{}".format(names[i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.title("Training Generator Loss")
    plt.xlabel("Epochs")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(base_path+"g_loss.png")
    plt.close()

    f = plt.figure()
    ax = plt.subplot(111)
    for i in range(len(discriminator_accuracies)):
        ax.plot(np.arange(len(discriminator_accuracies[i])), discriminator_accuracies[i], label="{}".format(names[i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.title("Training Discriminator Accuracy")
    plt.xlabel("Epochs")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(base_path+"d_acc_t.png")
    plt.close()

    if inception_scores is not None:
        f = plt.figure()
        ax = plt.subplot(111)
        for i in range(len(inception_scores)):
            ax.plot(np.arange(len(inception_scores[i])), inception_scores[i], label="{}".format(names[i]))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.title("Inception Score")
        plt.xlabel("Epochs")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        plt.ylabel("Inception Score")
        plt.grid(True)
        plt.savefig(base_path+"inception_score.png")
        plt.close()
