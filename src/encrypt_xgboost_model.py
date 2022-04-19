import pickle
from ppxgboost import BoosterParser, PaillierAPI, PPBooster
from ppxgboost.PPKey import PPBoostKey
from joblib import Parallel, delayed
import itertools
from secrets import token_bytes
from ope.pyope.ope import OPE
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from timerit import Timerit

from train_utils import load_dataset, train_test_split_undersample, convert_to_binary, show_time_taken, get_pickled_size

def chunk_arr(arr, num_chunks):
    chunk_size = len(arr) // num_chunks
    return [(arr[i:i+chunk_size]) for i in range(0,len(arr),chunk_size)]
    
def dechunk_arr(chunks):
    return list(itertools.chain.from_iterable(chunks))

def encrypt_model(model, min_max, num_cores):
    # load and parse model
    print('Encrypting model...')
    start_time = datetime.now()
    encrypted_model, _, _ = BoosterParser.model_to_trees(model, min_max)

    """
    Set up encryption materials:
        column_hash_key: for hashing the columns of the data
        paillier_public_key: for encrypting the leaf node values of the model trees
        paillier_private_key: for decrypting the output of the model
        order_preserving_key: for encryption/decryption of model inputs,
                              and comparison nodes in model trees
    """
    column_hash_key = token_bytes(16) # for hashing the columns of the data
    paillier_public_key, paillier_private_key = PaillierAPI.he_key_gen() #
    order_preserving_key = OPE(token_bytes(16))
    tree_encryption_keys = PPBoostKey(paillier_public_key, column_hash_key, order_preserving_key)

    output_chunks = Parallel(n_jobs=num_cores)(
        delayed(PPBooster.enc_xgboost_model)(
            tree_encryption_keys,
            chunk,
            PPBooster.MetaData(min_max)
        )
        for chunk in chunk_arr(encrypted_model, num_cores)
    )
    encrypted_model = dechunk_arr(output_chunks)

    show_time_taken('Time taken to encrypt model:', start_time, datetime.now())
    return encrypted_model, (column_hash_key, order_preserving_key, paillier_private_key, min_max)

def test_encrypted_model(plaintext_model, encrypted_model, keys, xtest, ytest, num_cores):
    print('\nTesting encrypted model...')
    start_time = datetime.now()
    start_index = ytest.index[0]
    test_idxs = np.array(ytest[ytest==1].index)
    test_idxs = np.random.choice(test_idxs, 75, replace = False) - start_index
    xtest = xtest.iloc[test_idxs, :]
    ytest = ytest.iloc[test_idxs]

    print('Making plaintext predictions.')
    preds_plaintext = convert_to_binary(plaintext_model.predict(xgb.DMatrix(xtest)))
    print('Plaintext predictions:', preds_plaintext)

    (column_hash_key, order_preserving_key, paillier_private_key, min_max) = keys
    encrypted_xtest = xtest.copy()
    print('Encrypting inputs.')
    encrypted_xtest = Parallel(n_jobs=num_cores)(
        delayed(PPBooster.enc_input_vector)(
            column_hash_key,
            order_preserving_key,
            xtest.columns,
            chunk,
            PPBooster.MetaData(min_max)
        )
        for chunk in chunk_arr(encrypted_xtest, num_cores)
    )
    encrypted_xtest = pd.concat(encrypted_xtest)

    print('Making encrypted predictions.')
    preds_encrypted = PPBooster.predict_binary(encrypted_model, encrypted_xtest)
    print('Decrypting results.')
    preds_decrypted = []
    for pred in preds_encrypted:
        preds_decrypted.append(PaillierAPI.decrypt(paillier_private_key, pred))
    preds_decrypted = convert_to_binary(np.array(preds_decrypted))
    print('Decrypted predictions:', preds_decrypted)

    result = np.array_equal(preds_plaintext, preds_decrypted)
    print('Success!') if result else print('Failed.')
    show_time_taken('Time taken for testing:', start_time, datetime.now())
    
    print('\nLatency Testing:')
    for _ in Timerit(num=5, label='Transaction Encryption', verbose=1):
        encrypted_xtest = xtest[:1].copy() # copying time was tested to be negligible
        PPBooster.enc_input_vector(column_hash_key, order_preserving_key, xtest.columns, encrypted_xtest, PPBooster.MetaData(min_max))

    for _ in Timerit(num=100, label='Plaintext Inference', verbose=1):
        convert_to_binary(plaintext_model.predict(xgb.DMatrix(xtest[:1])))

    for _ in Timerit(num=100, label='Encrypted Inference', verbose=1):
        PPBooster.predict_binary(encrypted_model, encrypted_xtest[:1])

    print('\nStorage size testing:')
    get_pickled_size(xtest[:1], 'Plaintext Transaction')
    get_pickled_size(encrypted_xtest, 'Encrypted Transaction')
    get_pickled_size(plaintext_model, 'Plaintext Model')
    get_pickled_size(encrypted_model, 'Encrypted Model')


if __name__ == '__main__':
    model_filename = dataset_name+'-xgboost.pt'
    model = pickle.load(open(model_filename, "rb"))
    # encrypted_model, keys = encrypt_model(model, 8)
    # pickle.dump(encrypted_model, open('encrypted-'+model_filename, "wb" ))
    # pickle.dump(keys, open('keys-'+model_filename, "wb" ))

    encrypted_model = pickle.load(open('encrypted-'+model_filename, "rb"))
    keys = pickle.load(open('keys-'+model_filename, "rb"))

    x, y = load_dataset('ulb')
    _, xtest, _, ytest = train_test_split_undersample(x, y, 0, test_size=0.25)

    result = test_encrypted_model(
        model, 
        encrypted_model,
        keys,
        xtest,
        ytest,
        8
    )
    print(result)