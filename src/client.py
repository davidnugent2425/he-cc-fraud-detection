import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import requests
from ppxgboost import PPBooster
from ppxgboost import PaillierAPI

from train_utils import load_dataset, convert_to_binary
from cc_he_utils import json_serialize, json_deserialize

if __name__ == '__main__':
    client_files_path = './client-files/'
    dataset_name = 'ulb'
    x, y = load_dataset(dataset_name)
    _, xtest, _, ytest = train_test_split(x, y, test_size=0.2, shuffle=False)

    test_input_index = np.random.randint(0, len(xtest))
    test_input = xtest.iloc[[test_input_index], :]
    correct_test_output = ytest.iloc[test_input_index]

    from IPython.display import display
    print('\nTest input:')
    display(test_input)
    print('\nExpected result:', correct_test_output)

    keys_path = client_files_path+'encrypted-{}-xgboost-keys.pt'.format(dataset_name)
    print('\nLoading keys from', keys_path)
    (column_hash_key, order_preserving_key, paillier_private_key, min_max) = pickle.load(
        open(keys_path, 'rb')
    )

    encrypted_input = test_input.copy()
    PPBooster.enc_input_vector(column_hash_key, order_preserving_key, set(xtest.columns), encrypted_input, PPBooster.MetaData(min_max))
    print('\nEncrypted input:')
    display(encrypted_input)

    request_data = {
        'input': json_serialize(encrypted_input)
    }
    server_endpoint = 'http://134.226.86.101:8000/xgboost/'+dataset_name
    print('\nSending encrypted input to', server_endpoint)
    response = requests.post(server_endpoint, json=request_data, timeout=10).json()
    encrypted_result = json_deserialize(response['result'])[0]
    print('\nReceived encrypted result:\n', encrypted_result)
    
    decrypted_result = PaillierAPI.decrypt(paillier_private_key, encrypted_result)
    decrypted_result = 1 if decrypted_result >= 0.5 else 0

    print('\nDecrypted result:', decrypted_result)
    if decrypted_result == correct_test_output:
        print('Which is correct!')
    else:
        print('Which is incorrect.')
