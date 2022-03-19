
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

def load_test_data():
    data = pd.read_csv('../data/creditcard.csv')
    # Preprocessing the data by scaling into a [0,1] range and splitting into inputs x and outputs y
    x = data.drop('Class', axis=1).values
    y = data['Class'].values
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # Splitting the data into a dataset for training the network and a dataset for testing it
    _, xtest, _, ytest = train_test_split(x, y, test_size=0.25, shuffle=False)
    return xtest, ytest

def json_serialize(_bytes):
    return pickle.dumps(_bytes).hex()

def json_deserialize(hex):
    return pickle.loads(bytearray.fromhex(hex))
