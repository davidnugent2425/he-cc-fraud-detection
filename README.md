# Privacy-Preserving Credit Card Fraud Detection using Homomorphic Encryption

This repo contains the source code of an implementation of a credit card fraud detector which uses private ML inference on encrypted transactions. A paper including the details the implementation is to be released soon.

The `src` folder contains the necessary source code. To run a simulation of the system, use the `Host.ipynb`, `Client.ipynb` and `Secure Middle Server.ipynb` Jupyter Notebooks.

## Environment Setup

To set up the environment required for running the scripts, the repository must be cloned including the [Privacy-Preserving XGBoost Inference](https://github.com/davidnugent2425/privacy-preserving-xgboost-inference) submodule.

```
git clone --recurse-submodules https://github.com/davidnugent2425/he-cc-fraud-detection.git
```

Then the relevant requirements must be installed. The commands below also set up a new virtual environment.

```
python3 -m venv ./env
source env/bin/activate
pip install -r requirements.txt
```

To set up a kernel which can be used to run the Jupyter Notebooks in this environment, run the following command

```
python -m ipykernel install --user --name=he-cc-fraud-detection
```

## Datasets Download

To download the relevant datasets, [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials) are a (free) prerequisite. The [rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) for the IEEE-CIS Fraud Detection must also be accepted (for the Vesta dataset). The following script can then be run which downloads the datasets into the appropriate directory

```
python download_datasets.py
```