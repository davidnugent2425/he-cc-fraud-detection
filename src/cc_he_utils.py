
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import tenseal as ts

dataset_path = '../data/creditcard.csv'

def load_test_data_np():
    xtest, ytest = load_test_data_df()
    xtest = xtest.values
    ytest = ytest.values
    scaler = MinMaxScaler()
    scaler.fit(xtest)
    xtest = scaler.transform(xtest)
    return xtest, ytest

def load_test_data_df():
    data = pd.read_csv(dataset_path)
    x = data.drop('Class', axis=1)
    y = data['Class']
    _, xtest, _, ytest = train_test_split(x, y, test_size=0.25, shuffle=False)
    return xtest, ytest

def json_serialize(_bytes):
    return pickle.dumps(_bytes).hex()

def json_deserialize(hex):
    return pickle.loads(bytearray.fromhex(hex))

def setup_tenseal_context():
    # TenSEAL CKKS HE encryption scheme context setup
    bits_scale = 50
    coeff_mod_bit_sizes = [60, bits_scale, bits_scale, bits_scale, 60]
    polynomial_modulus_degree = 8192*2

    # Create context
    context = ts.context(ts.SCHEME_TYPE.CKKS, polynomial_modulus_degree, coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    # Set global scale
    context.global_scale = 2**bits_scale
    # Generate galois keys required for matmul in ckks_vector
    context.generate_galois_keys()
    return context
