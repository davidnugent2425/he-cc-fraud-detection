import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

def load_dataset():
    data = pd.read_csv('../data/creditcard.csv')
    x = data.drop('Class', axis=1)
    y = data['Class']
    return x, y

def train_test_split_undersample(x, y, num_negatives, test_size=0.25):
    np.random.seed(0)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, shuffle=False)
    
    # Undersampling
    train_fraud_indices = np.array(ytrain[ytrain==1].index)
    train_normal_indices = np.array(ytrain[ytrain==0].index)
    train_random_normal_indices = np.random.choice(train_normal_indices, num_negatives, replace = False)
    train_indices = np.concatenate([train_fraud_indices, train_random_normal_indices])
    xtrain = xtrain.iloc[train_indices, :]
    ytrain = ytrain.iloc[train_indices]
    return xtrain, xtest, ytrain, ytest

def report_metrics(preds, truths):
    average_precision = metrics.average_precision_score(truths, preds)
    auc_roc = metrics.roc_auc_score(truths, preds)
    preds[preds>=0.5] = 1.0
    preds[preds<0.5] = 0.0
    conf_matrix = metrics.confusion_matrix(truths, preds)
    wandb.log({
        "average_precision": average_precision,
        "auc_roc": auc_roc,
        "false_positives": conf_matrix[0][1],
        "true_positives": conf_matrix[1][1],
    })