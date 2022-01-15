import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle

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
    df = pd.read_csv(dataset_csv_path, index_col='corporation')
    summary_dict = {}
    summary_dict['col_means'] = dict(df.mean())
    summary_dict['col_medians'] = dict(df.median())
    summary_dict['col_std'] = dict(df.std())
    return summary_dict

##################Function to get timings
#def execution_time():
    #calculate timing of training.py and ingestion.py
    #return #return a list of 2 timing values in seconds

##################Function to check dependencies
#def outdated_packages_list():
    #get a list of 


if __name__ == '__main__':
    model_predictions(data_path = test_data_path)
    dataframe_summary(data_path = dataset_csv_path)
    #execution_time()
    #outdated_packages_list()





    
