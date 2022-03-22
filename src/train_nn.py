import wandb
from train_utils import load_dataset, train_test_split_undersample, report_metrics
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

x, y = load_dataset()

config_defaults = {
    "undersampling_num_negatives": 853,
    "hidden_layer_size": 27,
    "pos_weight": 3,
}

sweep_config = {
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

def train():
    torch.manual_seed(0)
    np.random.seed(0)
    wandb.init(config=config_defaults, project="cc-he-nn")
    config = wandb.config

    xtrain, xtest, ytrain, ytest = train_test_split_undersample(x, y, config.undersampling_num_negatives, test_size=0.25)
    xtrain = xtrain.values
    ytrain = ytrain.values
    xtest = xtest.values
    ytest = ytest.values

    scaler = MinMaxScaler()
    scaler.fit(x.values)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)

    # Dataloaders setup, batch size of 100
    train_ds = torch.utils.data.TensorDataset(torch.tensor(xtrain).float(), torch.tensor(ytrain).float())
    test_ds = torch.utils.data.TensorDataset(torch.tensor(xtest).float(), torch.tensor(ytest).float())
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=100)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=100)

    model = Classifier(config.hidden_layer_size)
    wandb.watch(model, log_freq=100)

    pos_weight = torch.tensor([(config.undersampling_num_negatives/398)*config.pos_weight])
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n_epochs = 100

    for epoch in range(n_epochs):
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


if __name__ == '__main__':
    # sweep_id = wandb.sweep(sweep_config, project="cc-he-nn")
    # wandb.agent(sweep_id, train, count=50)
    train()