import wandb
import torch
import numpy as np
import pickle
from tqdm import tqdm
import time
from functools import partial
from train_utils import load_dataset, train_test_split_undersample, report_metrics, nn_configs, prep_data_nn, create_dataloaders, parse_training_args

# Plaintext Model
class Classifier(torch.nn.Module):
    def __init__(self, hidden_layer_size=15):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(30, hidden_layer_size)
        self.output_layer = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, x, show=False):
        self.show = show
        self.debug_output('Input', x)
        y = self.hidden_layer(x)
        self.debug_output('Hidden Layer', y)
        y = y * y
        self.debug_output('Activation', y)
        y = self.output_layer(y)
        self.debug_output('Output Layer', y)
        return y.squeeze()
    
    def debug_output(self, msg, vec):
        if self.show:
            print(msg)
            print(vec)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def train(x, y, configs, project_name, wandb_mode=None):
    print('Training neural network...')
    start_time = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    wandb.init(config=configs['defaults'], project=project_name, mode=wandb_mode)
    config = wandb.config

    xtrain, xtest, ytrain, ytest = train_test_split_undersample(x, y, config.undersampling_num_negatives, test_size=0.25)
    xtrain, xtest, ytrain, ytest = prep_data_nn(x, xtrain, xtest, ytrain, ytest)
    train_ds, test_ds, train_dl, test_dl = create_dataloaders(xtrain, xtest, ytrain, ytest)

    model = Classifier(config.hidden_layer_size)
    wandb.watch(model, log_freq=100)

    pos_weight = torch.tensor([(config.undersampling_num_negatives/398)*config.pos_weight])
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n_epochs = 100

    for epoch in tqdm(range(n_epochs)):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
            )
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        wandb.log({"test_loss": test_loss})
    
    preds = model(torch.tensor(xtest).float()).detach().numpy()
    report_metrics(preds, ytest)

    print('Time taken to train model:', (time.perf_counter()-start_time)/60)
    return model


if __name__ == '__main__':
    args = parse_training_args("Train a neural network compatible with conversion to a network using homomorphic encryption", ['ulb'])

    dataset_name = 'ulb' if not args['dataset'] else args['dataset']
    project_name = 'cc-{}-nn'.format(dataset_name)
    configs = nn_configs[dataset_name]

    x, y = load_dataset(dataset_name)
    wandb_mode = None if args['wandb'] else "disabled"

    if args['wandb_sweep']:
        sweep_id = wandb.sweep(configs['sweep'], project=project_name)
        wandb.agent(sweep_id, partial(train, x, y, configs, project_name), count=args['wandb_sweep'])

    else:
        model = train(x, y, configs, project_name, wandb_mode=wandb_mode)
        pickle.dump(model, open(dataset_name+'-nn.pt', "wb" ))