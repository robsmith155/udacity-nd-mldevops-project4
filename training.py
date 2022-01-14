from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

################### Load config.json and get path variables ##################
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl') 

####################### Load and prepare data ################################

df = pd.read_csv(dataset_csv_path, index_col='corporation')
y = df['exited'].values.reshape(-1,1).ravel()
X = df.drop(columns='exited').values.reshape(-1,3)
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=155)

#################Function for training the model
def train_model() -> None:
    """
    Function to train a Logistic Regression classifier
    """
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, 
                    fit_intercept=True, intercept_scaling=1, l1_ratio=None, 
                    max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(X_train, y_train)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model()