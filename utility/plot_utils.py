from keras.utils.vis_utils import plot_model
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import os
import itertools


def plot_and_save_model(model, filename):
    plot_model(model, show_shapes=True, to_file=filename)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    """
    See: https://stackoverflow.com/a/26980472

    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show. (Note: current implementation merely prints and does not
    return top classes.)
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


def plot_history_2win(history):
    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color='g', label='Train')
    plt.plot(history.history['val_acc'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='g', label='Train')
    plt.plot(history.history['val_loss'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def create_history_plot(history, model_name):
    plt.title('Accuracy and Loss (' + model_name + ')')
    plt.plot(history.history['acc'], color='g', label='Train Accuracy')
    plt.plot(history.history['val_acc'], color='b', label='Validation Accuracy')
    plt.plot(history.history['loss'], color='r', label='Train Loss')
    plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')

    plt.tight_layout()


def plot_history(history, model_name):
    create_history_plot(history, model_name)
    plt.show()


def plot_and_save_history(history, model_name, file_path):
    create_history_plot(history, model_name)
    plt.savefig(file_path)


## Precision-Recall Curve function for framenet clusters
def df_for_plot(true_labels):
    '''
    Make a dataframe to draw a ROC curve
    '''
    df = pd.DataFrame(true_labels, columns=['True'])
    df['Pred'] = 0
    return df

def plot_sub_prcuve(df, fig, axes, frame_num, x=0, y=0):
    trues = np.array(df['True_bin'])  # Answers are binary(Right:1, Wrong:0)
    prds = np.array(df['pred_percent'])  # Get predicted percentage

    precision, recall, _ = precision_recall_curve(trues, prds)  # Get precision_recall
    # plot the precision-recall curves
    no_skill = len(trues[trues==1]) / len(trues)  # base value

    axes[x, y].set_title('Label: {}'.format(frame_num))
    axes[x, y].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    axes[x, y].plot(recall, precision, marker='.', label='BERT-base')
    # axis labels
    axes[x, y].set_xlabel('Recall')
    axes[x, y].set_ylabel('Precision')

def plot_prcurve(model_name, true_labels, predictions):
    df = df_for_plot(true_labels)

    x, y = 0, 0
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))  # Drawing 7 graphs for our case
    for i in range(7):
        df['True_bin'] = df['True'].apply(lambda x: 1 if (x ==i) else 0)  # Substitute answers as binary values
        result = []
        for j in range(len(predictions)):
            result.append(predictions[j][i])  # Add answer probability column
        df['pred_percent'] = result
        
        plot_sub_prcuve(df, fig, axes, i, x, y)  # precision-recall curve
        plt.tight_layout()
        # Locate subplot orientation
        if y < 3:
            y += 1
        elif x == 1 and y ==3:
            break
        else:
            x += 1
            y = 0
    
    if not os.path.exists('./images'):
        os.mkdir('./images')

    plt.savefig(f'./images/{model_name}_prcurve.jpg')
    plt.show()