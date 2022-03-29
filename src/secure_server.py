import pickle
import argparse
from encrypt_xgboost_model import encrypt_model

parser = argparse.ArgumentParser(
        description='script for encrypting xgboost model for inference using homomorphic encryption',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument("-c", "--cores", metavar='NUM-CORES', type=int, default=1, help="set number of cores for paralellization")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    server_files_path = './server-files/'
    client_files_path = './client-files/'

    for dataset_name in ['ulb', 'ieee']:
        print('\nLoading model for {} dataset'.format(dataset_name))
        model_filename = server_files_path+'{}-xgboost.pt'.format(dataset_name)
        (model, min_max) = pickle.load(open(model_filename, "rb"))

        encrypted_model, keys = encrypt_model(model, min_max, args['cores'])

        encrypted_model_filename = server_files_path+'encrypted-{}-xgboost.pt'.format(dataset_name)
        print('Saving encrypted model to', encrypted_model_filename)
        pickle.dump(encrypted_model, open(encrypted_model_filename, "wb" ))

        keys_filename = client_files_path+'encrypted-{}-xgboost-keys.pt'.format(dataset_name)
        print('Saving keys to', keys_filename)
        pickle.dump(keys, open(keys_filename, "wb" ))