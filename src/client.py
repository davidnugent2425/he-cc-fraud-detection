import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import requests
from ppxgboost import PPBooster
from ppxgboost import PaillierAPI
import argparse
from IPython.display import display
import tenseal as ts
from sklearn import preprocessing
from datetime import datetime

from train_utils import load_dataset, convert_to_binary, prep_data_nn, show_time_taken
from cc_he_utils import json_serialize, json_deserialize, setup_tenseal_context

parser = argparse.ArgumentParser(
    description='script for sending encrypted input data to fraud detection server',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-f", "--fraudulant", action="store_true", help="ensure test datapoint used is fraudulant")
parser.add_argument("-d", "--dataset", choices=['ulb', 'ieee'], default='ulb', help="name of dataset to test with: ulb or ieee")
parser.add_argument("-m", "--model", choices=['nn', 'xgboost'], default='xgboost', help="name of model to use as detector: nn or xgboost")


if __name__ == '__main__':
    args = vars(parser.parse_args())
    client_files_path = './client-files/'
    server_url = 'http://134.226.86.101:8000'
    dataset_name = args['dataset']
    x, y = load_dataset(dataset_name)
    _, xtest, _, ytest = train_test_split(x, y, test_size=0.2, shuffle=False)

    if args['fraudulant']:
        xtest = xtest[ytest==1]
        ytest = ytest[ytest==1]

    test_input_index = np.random.randint(0, len(xtest))
    test_input = xtest.iloc[[test_input_index], :]
    correct_test_output = ytest.iloc[test_input_index]

    print('\nTest input:')
    display(test_input)
    print('\nExpected result:', correct_test_output)

    if args['model'] == 'xgboost':
        keys_path = client_files_path+'encrypted-{}-xgboost-keys.pt'.format(dataset_name)
        print('\nLoading keys from', keys_path)
        (column_hash_key, order_preserving_key, paillier_private_key, min_max) = pickle.load(
            open(keys_path, 'rb')
        )

        encrypted_input = test_input.copy()
        start_time = datetime.now()
        PPBooster.enc_input_vector(column_hash_key, order_preserving_key, set(xtest.columns), encrypted_input, PPBooster.MetaData(min_max))
        print('\nEncrypted input:')
        display(encrypted_input)

        request_data = {
            'input': json_serialize(encrypted_input)
        }
        server_endpoint = server_url+'/xgboost/'+dataset_name
        print('\nSending encrypted input to', server_endpoint)
        response = requests.post(server_endpoint, json=request_data, timeout=10).json()
        encrypted_result = json_deserialize(response['result'])[0]
        print('\nReceived encrypted result:\n', encrypted_result)
        
        decrypted_result = PaillierAPI.decrypt(paillier_private_key, encrypted_result)

    elif args['model'] == 'nn':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(x.values)
        test_input = test_input.values
        test_input = scaler.transform(test_input)[0]

        context = setup_tenseal_context()
        start_time = datetime.now()
        encrypted_input = ts.ckks_vector(context, test_input)
        print('\nEncrypted input:')
        print(encrypted_input)

        request_data = {
            'input': json_serialize(encrypted_input.serialize()),
            'context': json_serialize(context.serialize()),
        }
        server_endpoint = server_url+'/nn/'+dataset_name
        print('\nSending encrypted input to', server_endpoint)
        response = requests.post(server_endpoint, json=request_data).json()
        encrypted_result = ts.ckks_vector_from(context, json_deserialize(response['result']))
        print('\nReceived encrypted result:\n', encrypted_result)
        
        decrypted_result = encrypted_result.decrypt()[0]
    
    decrypted_result = 1 if decrypted_result >= 0.5 else 0
    print('\nDecrypted result:', decrypted_result)
    if decrypted_result == correct_test_output:
        print('Which is correct!')
    else:
        print('Which is incorrect.')

    show_time_taken('Time taken from encryption of input to decryption of result:', start_time, datetime.now())
