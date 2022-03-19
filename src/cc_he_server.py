from flask import Flask, request, send_file
import os
import tenseal as ts
import torch
import pickle
from cc_he_utils import json_serialize, json_deserialize, load_test_data

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

hidden_layer = torch.load('./nn-hidden-layer.pt')
output_layer = torch.load('./nn-output-layer.pt')
he_model = HEModel(hidden_layer, output_layer)

app = Flask(__name__)

@app.route('/nn', methods=['POST'])
def infer_nn():
    print(len(request.data))
    data = request.get_json()
    context = ts.context_from(json_deserialize(data['context']))
    enc_input = ts.ckks_vector_from(context, json_deserialize(data['input']))
    result = he_model(enc_input)
    return {
        'result': json_serialize(result.serialize())
    }

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8000
    )