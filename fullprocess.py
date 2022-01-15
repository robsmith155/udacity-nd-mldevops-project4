import ast
import json
import os
import pandas as pd
import subprocess

from utils import load_data
from scoring import score_model

################### Load config.json and get path variables ##################
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
ingestedfiles_path = os.path.join(config['output_folder_path'], 'ingestedfiles.txt')
ingesteddata_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
lastestscore_path = os.path.join(config['prod_deployment_path'], 'latestscore.txt')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl') 

##################Check and read new data
#first, read ingestedfiles.txt
with open(ingestedfiles_path, 'r') as f:
    ingested_files = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(input_folder_path)
new_csv_files = []
    
for file in filenames:
    if file.endswith('.csv'):
        if os.path.basename(file) not in ingested_files:
            new_csv_files.append(file)
    else:
        pass


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_csv_files) > 0:
    subprocess.run(['python3', 'ingestion.py'])

#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(lastestscore_path, 'r') as f:
    latest_score = float(f.read())

score = score_model(data_path=ingesteddata_path)

model_drift = score < latest_score

if model_drift == False:
    print(f'Model drift did not occur. Previous model F1 score was {latest_score}. New model score is {score}. Ending here.')
else:
    # Retrain and redeploy model
    print(f'Model drift occurred. Previous model F1 score was {latest_score}. New model score is {score}. Training new model.')
    
    # Retrain model with latest data
    subprocess.run(['python3', 'training.py'])

    # Score model on test data
    subprocess.run(['python3', 'scoring.py'])

    # Redeploy model
    subprocess.run(['python3', 'deployment.py'])

    # Generate report
    subprocess.run(['python3', 'reporting.py'])

    # Run diagnostics
    subprocess.run(['python3', 'apicalls.py'])






