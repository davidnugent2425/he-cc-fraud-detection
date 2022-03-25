import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics, preprocessing
import argparse
import torch
from seaborn import heatmap
from matplotlib import pyplot as plt

def load_dataset(name='ulb'):
    print('Loading {} dataset...'.format(name))
    if name == 'ulb':
        data = pd.read_csv('../data/creditcard.csv')
        x = data.drop('Class', axis=1)
        y = data['Class']

    elif name == 'ieee':
        transactions = pd.read_csv('../data/train_transaction.csv', index_col='TransactionID')
        identities = pd.read_csv('../data/train_identity.csv', index_col='TransactionID')
        data = transactions.merge(identities, how='left', left_index=True, right_index=True)
        
        x = data.drop('isFraud', axis=1)
        y = data['isFraud']
        x = x.fillna(-999)

        for f in x.columns:
            if x[f].dtype=='object': 
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(x[f].values))
                x[f] = lbl.transform(list(x[f].values))

    print('Dataset loaded!')
    return x, y

def train_test_split_undersample(x, y, num_negatives, test_size=0.25):
    np.random.seed(0)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, shuffle=False)
    
    # Undersampling
    train_fraud_indices = np.array(ytrain[ytrain==1].index)
    train_normal_indices = np.array(ytrain[ytrain==0].index)
    train_random_normal_indices = np.random.choice(train_normal_indices, num_negatives, replace = False)
    start_index = xtrain.index[0]
    train_indices = np.concatenate([train_fraud_indices, train_random_normal_indices]) - start_index
    xtrain = xtrain.iloc[train_indices, :]
    ytrain = ytrain.iloc[train_indices]
    return xtrain, xtest, ytrain, ytest
    
def summarize_data(y, name):
    print('{}:'.format(name), len(y), 'total values, num fraudulant:', len(y[y==1].index))

def prep_data_nn(x, y, scaler):
    x = x.values
    y = y.values
    x = scaler.transform(x)
    return x, y

def create_dataloader(x, y):
    ds = torch.utils.data.TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=100)
    return dl

def convert_to_binary(preds, threshold=0.5):
    preds[preds>=0.5] = 1.0
    preds[preds<0.5] = 0.0
    return preds

def report_metrics(preds, truths):
    average_precision = metrics.average_precision_score(truths, preds)
    auc_roc = metrics.roc_auc_score(truths, preds)
    preds = convert_to_binary(preds)
    conf_matrix = metrics.confusion_matrix(truths, preds)
    wandb.log({
        "average_precision": average_precision,
        "auc_roc": auc_roc,
        "false_positives": conf_matrix[0][1],
        "true_positives": conf_matrix[1][1],
    })

def evaluate_predictions(preds, truths):
    print('\nTest results:')
    print('Average precision: {}'.format(metrics.average_precision_score(truths, preds)))
    print('AUC ROC score: {}'.format(metrics.roc_auc_score(truths, preds)))
    preds = convert_to_binary(preds)
    print('Accuracy score: {}'.format(metrics.accuracy_score(truths, preds)))
    print(metrics.classification_report(truths, preds))

    conf_matrix = metrics.confusion_matrix(truths, preds)
    plt.figure(figsize=(5, 5))
    labels = ["Valid", "Fraud"]
    heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def parse_training_args(description, datasets):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-d", "--dataset", choices=datasets, help="name of dataset to train on: ulb or ieee")
    parser.add_argument("-w", "--wandb", action="store_true", help="log training using Weights & Biases")
    parser.add_argument("-ws", "--wandb-sweep", metavar='NUM-RUNS', type=int, help="run sweep on training hyperparameters using Weights & Biases")
    parser.add_argument("-c", "--cores", metavar='NUM-CORES', type=int, default=1, help="set number of cores for paralellization")
    return vars(parser.parse_args())

xgboost_configs = {
    "ulb": {
        "defaults": {
            "max_depth": 7,
            "learning_rate": 0.2,
            "undersampling_num_negatives": 2150,
            "num_estimators": 89,
        },

        "sweep": {
            "method": "bayes",
            "metric": {
            "name": "average_precision",
            "goal": "maximize"   
            },
            "parameters": {
        #         "booster": {
        #             "values": ["gbtree", "gblinear"]
        #         },
                "max_depth": {
                    "min": 6,
                    "max": 12,
                },
                "learning_rate": {
                    "values": [0.1, 0.2]
                },
                "undersampling_num_negatives": {
                    "min": 1500,
                    "max": 2500,
                },
                "num_estimators": {
                    "min": 50,
                    "max": 100,
                },
            }
        }
    },

    "ieee": {
        "defaults": {
            "max_depth": 7,
            "learning_rate": 0.2,
            "undersampling_num_negatives": 15565,
            "num_estimators": 89,
        }
    }
}

nn_configs = {
    "ulb": {
        "defaults": {
            "undersampling_num_negatives": 853,
            "hidden_layer_size": 27,
            "pos_weight": 3,
        },

        "sweep": {
            "method": "random",
            "metric": {
            "name": "average_precision",
            "goal": "maximize"   
            },
            "parameters": {
                "undersampling_num_negatives": {
                    "min": 500,
                    "max": 1500,
                },
                "hidden_layer_size": {
                    "min": 20,
                    "max": 60,
                },
                "pos_weight": {
                    "min": 1,
                    "max": 5,
                }
            }
        }
    }
}

