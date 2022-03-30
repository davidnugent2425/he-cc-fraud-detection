from flask import Flask, request, send_file
import os
import tenseal as ts
import torch
import pickle
import xgboost as xgb
from ppxgboost import PPBooster as ppbooster
from train_nn import Classifier
from IPython.display import display
from datetime import datetime
from cc_he_utils import json_serialize, json_deserialize
from train_utils import show_time_taken

# HE Model
class HEModel:
    def __init__(self, hidden_layer, output_layer):
        self.hidden_layer_weight = hidden_layer.weight.t().tolist()
        self.hidden_layer_bias = hidden_layer.bias.tolist()
        self.output_layer_weight = output_layer.weight.t().tolist()
        self.output_layer_bias = output_layer.bias.tolist()
        
    def forward(self, enc_x, show=False, plaintext=False):
        self.show = show
        self.plaintext = plaintext
        self.debug_output('Input', enc_x)
        enc_y = enc_x.mm(self.hidden_layer_weight) + self.hidden_layer_bias
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

server_files_path = './server-files/'
ulb_plaintext_nn_model = pickle.load(open(server_files_path+'ulb-nn.pt', "rb"))
ulb_encrypted_nn_model = HEModel(ulb_plaintext_nn_model.hidden_layer, ulb_plaintext_nn_model.output_layer)

ulb_encrypted_xgboost_model = pickle.load(open(server_files_path+'encrypted-ulb-xgboost.pt', "rb"))
ieee_encrypted_xgboost_model = pickle.load(open(server_files_path+'encrypted-ieee-xgboost.pt', "rb"))

app = Flask(__name__)

@app.route('/nn/ulb', methods=['POST'])
def infer_nn():
    data = request.get_json()
    context = ts.context_from(json_deserialize(data['context']))
    enc_input = ts.ckks_vector_from(context, json_deserialize(data['input']))
    print('\nReceived encrypted input for ulb neural network model:')
    print(enc_input)
    start_time = datetime.now()
    result = ulb_encrypted_nn_model(enc_input)
    show_time_taken('Inference time:', start_time, datetime.now())
    return {
        'result': json_serialize(result.serialize())
    }

@app.route('/xgboost/ulb', methods=['POST'])
def infer_xgboost_ulb():
    data = request.get_json()
    enc_input = json_deserialize(data['input'])
    print('\nReceived encrypted input for ulb xgboost model:')
    display(enc_input)
    start_time = datetime.now()
    result = ppbooster.predict_binary(ulb_encrypted_xgboost_model, enc_input)
    show_time_taken('Inference time:', start_time, datetime.now())
    return {
        'result': json_serialize(result)
    }

@app.route('/xgboost/ieee', methods=['POST'])
def infer_xgboost_ieee():
    data = request.get_json()
    enc_input = json_deserialize(data['input'])
    print('\nReceived encrypted input for ieee xgboost model:')
    display(enc_input)
    start_time = datetime.now()
    result = ppbooster.predict_binary(ieee_encrypted_xgboost_model, enc_input)
    show_time_taken('Inference time:', start_time, datetime.now())
    return {
        'result': json_serialize(result)
    }

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8000
    )