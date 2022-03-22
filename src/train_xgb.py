import wandb
import xgboost as xgb
from wandb.xgboost import WandbCallback
from xgboost import XGBClassifier
from train_utils import load_dataset, train_test_split_undersample, report_metrics

x, y = load_dataset()

config_defaults = {
    "max_depth": 7,
    "learning_rate": 0.2,
    "undersampling_num_negatives": 2150,
    "num_estimators": 89,
}

sweep_config = {
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

def train():
    wandb.init(config=config_defaults, project="cc-he-xgboost")
    config = wandb.config

    xtrain, xtest, ytrain, ytest = train_test_split_undersample(x, y, config.undersampling_num_negatives, test_size=0.25)
    
    clf = xgb.XGBClassifier(
        booster="gbtree",
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        num_estimators=config.num_estimators,
        tree_method='gpu_hist',
    )
    clf.fit(xtrain, ytrain)
    
    model = clf.get_booster()
    preds = model.predict(xgb.DMatrix(xtest))
    report_metrics(preds, ytest)
    
    return model

if __name__ == '__main__':
    # sweep_id = wandb.sweep(sweep_config, project="cc-he-xgboost")
    # wandb.agent(sweep_id, train, count=50)
    train()