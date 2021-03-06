import wandb
import torch
import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime
import tenseal as ts
from timerit import Timerit
from train_utils import load_dataset, train_test_split_undersample, report_metrics, nn_configs, prep_data_nn, create_dataloader, parse_training_args, evaluate_predictions, show_time_taken, convert_to_binary, get_pickled_size
from cc_he_utils import setup_tenseal_context

def test_encrypted_nn(plaintext_model, xtest, ytest, scaler):
    print('\nTesting encrypted model...')
    start_time = datetime.now()
    start_index = ytest.index[0]
    # test_idxs = np.array(ytest[ytest==1].index)
    # test_idxs = np.random.choice(test_idxs, 75, replace = False) - start_index
    test_fraud_idxs = np.array(ytest[ytest==1].index)
    test_fraud_idxs = np.random.choice(test_fraud_idxs, 75, replace = False)
    test_normal_idxs = np.array(ytest[ytest==0].index)
    test_normal_idxs = np.random.choice(test_normal_idxs, 75, replace = False)
    test_idxs = np.concatenate([test_fraud_idxs, test_normal_idxs]) - start_index
    xtest = xtest.iloc[test_idxs, :]
    ytest = ytest.iloc[test_idxs]
    xtest, ytest = prep_data_nn(xtest, ytest, scaler)

    print('Making plaintext predictions.')
    preds_plaintext = convert_to_binary(plaintext_model(torch.tensor(xtest).float()).detach().numpy())
    print('Plaintext predictions:', preds_plaintext)
    print(len(preds_plaintext))

    num_multiplications = len(plaintext_model.hidden_layers)*2 + 1
    context = setup_tenseal_context(multiplicative_depth=num_multiplications)
    encrypted_model = HEModel(plaintext_model.hidden_layers, plaintext_model.output_layer)
    print('Encrypting inputs, making encrypted predictions, and decrypting.')
    preds_decrypted = []
    for row in xtest:
        encrypted_input = ts.ckks_vector(context, row)
        encrypted_pred = encrypted_model(encrypted_input)
        decrypted_pred = encrypted_pred.decrypt()
        preds_decrypted.append(decrypted_pred)
    preds_decrypted = convert_to_binary(np.array(preds_decrypted).squeeze())
    print('Decrypted predictions:', preds_decrypted)

    result = np.array_equal(preds_plaintext, preds_decrypted)
    print('Success!') if result else print('Failed.')

    print('\nLatency Testing:')
    for _ in Timerit(num=100, label='Transaction Encryption', verbose=1):
        encrypted_input = ts.ckks_vector(context, xtest[0])

    for _ in Timerit(num=100, label='Plaintext Inference', verbose=1):
        convert_to_binary(plaintext_model(torch.tensor(xtest[0]).float()).detach().numpy())

    for _ in Timerit(num=100, label='Encrypted Inference', verbose=1):
        encrypted_model(encrypted_input)

    print('\nStorage size testing:')
    get_pickled_size(xtest[0], 'Plaintext Transaction')
    get_pickled_size(encrypted_input.serialize(), 'Encrypted Transaction')
    get_pickled_size(plaintext_model, 'Plaintext Model')

# HE Model
class HEModel:
    def __init__(self, hidden_layers, output_layer):
        self.hidden_layer_weights = [hidden_layer.weight.t().tolist() for hidden_layer in hidden_layers]
        self.hidden_layer_biases = [hidden_layer.bias.tolist() for hidden_layer in hidden_layers]
        self.output_layer_weight = output_layer.weight.t().tolist()
        self.output_layer_bias = output_layer.bias.tolist()
        
    def forward(self, enc_y, show=False, plaintext=False):
        self.show = show
        self.plaintext = plaintext
        self.debug_output('Input', enc_y)
        for i in range(len(self.hidden_layer_weights)):
            enc_y = enc_y.mm(self.hidden_layer_weights[i]) + self.hidden_layer_biases[i]
            self.debug_output('Hidden Layer', enc_y)
            enc_y *= enc_y
            self.debug_output('Activation', enc_y)
        enc_y = enc_y.mm(self.output_layer_weight) + self.output_layer_bias
        self.debug_output('Output Layer', enc_y)
        return enc_y
    
    def debug_output(self, msg, vec):
        if self.show:
            print(msg)
            if self.plaintext: vec = torch.tensor(vec.decrypt())
            print(vec)
    
    def __call__(self, enc_x, show=False, plaintext=False):
        return self.forward(enc_x, show, plaintext)

# Plaintext Model
class Classifier(torch.nn.Module):
    def __init__(self, hidden_layer_size=15, input_size=30, num_hidden_layers=1):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_layer_size)])
        self.hidden_layers.extend([torch.nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(num_hidden_layers-1)])
        self.output_layer = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, y, show=False):
        self.show = show
        self.debug_output('Input', y)
        for hidden_layer in self.hidden_layers:
            y = hidden_layer(y)
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

def train(x, y, configs, project_name, wandb_mode=None, n_epochs=100, lr=0.001): # 0.00001 for Vesta
    print('Training neural network...')
    start_time = datetime.now()
    torch.manual_seed(0)
    np.random.seed(0)
    wandb.init(config=configs['defaults'], project=project_name, mode=wandb_mode)
    config = wandb.config

    # n_epochs = 300
    # lr = 0.00001

    # split data into data used for training and data used for testing
    xtrain, xtest_df, ytrain, ytest_df = train_test_split(x, y, test_size=0.2, shuffle=False)
    # split training data into training and validation sets for hyperparameter tuning
    xtrain, xvalid, ytrain, yvalid = train_test_split_undersample(xtrain, ytrain, config.undersampling_num_negatives, test_size=0.1875)

    num_columns = len(xtrain.columns)

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x.values)

    xtrain, ytrain = prep_data_nn(xtrain, ytrain, scaler)
    xvalid, yvalid = prep_data_nn(xvalid, yvalid, scaler)
    xtest, ytest = prep_data_nn(xtest_df, ytest_df, scaler)

    train_dl = create_dataloader(xtrain, ytrain)
    valid_dl = create_dataloader(xvalid, yvalid)

    model = Classifier(config.hidden_layer_size, num_columns, num_hidden_layers=1)
    wandb.watch(model, log_freq=100)

    pos_weight = torch.tensor([(config.undersampling_num_negatives/398)*config.pos_weight])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in tqdm(range(n_epochs)):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        wandb.log({"validation_loss": test_loss})
    
    preds = model(torch.tensor(xvalid).float()).detach().numpy()
    report_metrics(preds, yvalid)

    show_time_taken('Time taken to train model:', start_time, datetime.now())
    # model = pickle.load(open('./server-files/ulb-nn.pt', "rb"))

    test_encrypted_nn(model, xtest_df, ytest_df, scaler)

    test_preds = model(torch.tensor(xtest).float()).detach().numpy()
    evaluate_predictions(test_preds, ytest)

    return model


if __name__ == '__main__':
    args = parse_training_args("Train a neural network compatible with conversion to a network using homomorphic encryption", ['ulb', 'ieee'])

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
        filepath = './server-files/'+dataset_name+'-nn.pt'
        print('\nSaving plaintext neural network model at', filepath)
        pickle.dump(model, open(filepath, "wb" ))