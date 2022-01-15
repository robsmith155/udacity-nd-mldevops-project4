import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

from utils import load_data

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 

##################Function to get model predictions
def model_predictions(data_path: str) -> list:
    """
    Function to compute predictions on data in data_path.

    Inputs
    ------
    data_path : str
        Path to dataset to run predictions on.
    Returns
    -------
    y_pred : list
        Model predictions.
    """
    X, y = load_data(data_path = data_path)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    return y_pred.tolist()


##################Function to get summary statistics
def dataframe_summary(data_path: str) -> dict:
    """
    Function to calculate summary statistics on the training dataset.

    Inputs
    ------
    data_path : str
        Filepath of dataset to summarize.
    Returns
    -------
    summary_dict : dict
        Dictionary of the dataset summary statistics.
    """
    df = pd.read_csv(data_path, index_col='corporation')
    summary_dict = {}
    summary_dict['col_means'] = dict(df.mean())
    summary_dict['col_medians'] = dict(df.median())
    summary_dict['col_std'] = dict(df.std())
    return summary_dict


#################### Function to check missing data ###################

def missing_data(data_path) -> list:
    """
    Function to count missing values in each column. Output in percent.

    Inputs
    ------
    data_path : str
        Filepath of dataset to count missing data.
    Returns
    -------
    col_na_pc : list
        List of column missing values in percent.
    """
    df = pd.read_csv(data_path, index_col='corporation')
    col_na_count = list(df.isna().sum())
    col_na_pc = [col_na_count[i]/len(df) for i in range(len(col_na_count))]
    return col_na_pc


###################### Function to get timings ##############################

def execution_time() -> list:
    """
    Function to compute the time to run the ingestion.py and training.py 
    scripts.

    Inputs
    ------
    None

    Returns
    -------
    runtimes : list
        List of runtimes for ingestion and training (in seconds).
    """

    # Time the ingestion.py script
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_runtime = timeit.default_timer() - starttime
    
    # Time the training.py script
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_runtime = timeit.default_timer() - starttime
    
    runtimes = [ingestion_runtime, training_runtime]

    return runtimes


###################### Function to check dependencies #######################

def outdated_packages_list() -> list:
    """
    Compare installed Python packages and output list of packages which are 
    not using the latest version. This is a list of dictionaries which includes
    the package name, currently installed version and the latest availavle
    version.

    Inputs
    ------
    None

    Returns
    -------
    runtimes : list
        List of runtimes for ingestion and training (in seconds).
    """
    outdated = subprocess.run(['pip', 'list', '--outdated', '--format', 'json'], capture_output=True).stdout
    outdated = outdated.decode('utf8').replace("'", '"')
    outdated_list = json.loads(outdated)
    return outdated_list


if __name__ == '__main__':
    model_predictions(data_path=test_data_path)
    dataframe_summary(data_path=dataset_csv_path)
    missing_data(data_path=dataset_csv_path)
    execution_time()
    outdated_packages_list()





    
