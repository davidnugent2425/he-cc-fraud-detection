import wandb
import xgboost as xgb
from wandb.xgboost import WandbCallback
from xgboost import XGBClassifier
import pickle
from encrypt_xgboost_model import encrypt_model, test_encrypted_model
from ppxgboost import BoosterParser
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from functools import partial
from datetime import datetime

from train_utils import load_dataset, train_test_split_undersample, report_metrics, xgboost_configs, parse_training_args, summarize_data, evaluate_predictions, show_time_taken


def train(x, y, configs, project_name, num_cores=1, test_encrypted=False, wandb_mode=None):
    start_time = datetime.now()

    wandb.init(config=configs['defaults'], project=project_name, mode=wandb_mode)
    config = wandb.config

    # split data into data used for training and data used for testing
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=False)
    # split training data into training and validation sets for hyperparameter tuning
    xtrain, xvalid, ytrain, yvalid = train_test_split_undersample(xtrain, ytrain, config.undersampling_num_negatives, test_size=0.1875)
    
    summarize_data(ytrain, 'training set')
    summarize_data(yvalid, 'valid set')
    summarize_data(ytest, 'test set')
    
    print('\nTraining XGBoost model...')
    clf = xgb.XGBClassifier(
        booster='gbtree',
        max_depth=config.max_depth,
        learning_rate=0.2,
        n_estimators=config.num_estimators,
        tree_method='gpu_hist',
        eval_metric='logloss',
        use_label_encoder=False
    )
    clf.fit(xtrain, ytrain)
    
    model = clf.get_booster()
    preds = model.predict(xgb.DMatrix(xvalid))
    report_metrics(preds, yvalid)
    
    show_time_taken('Time taken to train model:', start_time, datetime.now())

    # min and max values of the dataset must be saved for model encryption
    min_max = BoosterParser.training_dataset_parser(x)

    if test_encrypted:
        encrypted_model, keys = encrypt_model(model, min_max, num_cores)
        test_encrypted_model(
            model, 
            encrypted_model,
            keys,
            xtest,
            ytest,
            num_cores
        )

    test_preds = model.predict(xgb.DMatrix(xtest))
    evaluate_predictions(test_preds, ytest)

    return (model, min_max)


if __name__ == '__main__':
    args = parse_training_args("Train an xgboost model and encrypt it to be compatible with homomorphic encryption.", ['ulb', 'ieee'])

    dataset_name = 'ulb' if not args['dataset'] else args['dataset']
    project_name = 'cc-{}-xgboost'.format(dataset_name)
    configs = xgboost_configs[dataset_name]
    wandb_mode = None if args['wandb'] else "disabled"

    x, y = load_dataset(dataset_name)
    summarize_data(y, dataset_name)

    if args['wandb_sweep']:
        sweep_id = wandb.sweep(configs['sweep'], project=project_name)
        wandb.agent(sweep_id, partial(train, x, y, configs, project_name, args['cores']), count=args['wandb_sweep'])

    else:
        model = train(x, y, configs, project_name, args['cores'], test_encrypted=True, wandb_mode=wandb_mode)
        filepath = './server-files/'+dataset_name+'-xgboost.pt'
        print('\nSaving plaintext xgboost model at', filepath)
        pickle.dump(model, open(filepath, "wb"))