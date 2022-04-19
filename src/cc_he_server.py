from flask import Flask, request, send_file
import os
import tenseal as ts
import torch
import pickle
import xgboost as xgb
from ppxgboost import PPBooster as ppbooster
from train_nn import Classifier, HEModel
from IPython.display import display
from datetime import datetime
from cc_he_utils import json_serialize, json_deserialize
from train_utils import show_time_taken


server_files_path = './server-files/'
ulb_plaintext_nn_model = pickle.load(open(server_files_path+'ulb-nn.pt', "rb"))
ulb_encrypted_nn_model = HEModel(ulb_plaintext_nn_model.hidden_layers, ulb_plaintext_nn_model.output_layer)

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