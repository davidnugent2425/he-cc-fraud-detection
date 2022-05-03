from kaggle.api.kaggle_api_extended import KaggleApi
import os
from zipfile import ZipFile

if __name__ == '__main__':
    os.mkdir('./data')
    os.chdir('./data')
    api = KaggleApi()
    api.authenticate()

    print('Downloading ULB dataset...')
    api.dataset_download_files('mlg-ulb/creditcardfraud')
    ulb_zip_path = 'creditcardfraud.zip'
    with ZipFile(ulb_zip_path, 'r') as zipObj:
        zipObj.extractall()
    os.remove(ulb_zip_path)
    
    print('Downloading Vesta dataset...')
    api.competition_download_files('ieee-fraud-detection')
    vesta_zip_path = 'ieee-fraud-detection.zip'
    with ZipFile(vesta_zip_path, 'r') as zipObj:
        zipObj.extractall()
    os.remove(vesta_zip_path)
    os.remove('test_transaction.csv')
    os.remove('test_identity.csv')
    os.remove('sample_submission.csv')