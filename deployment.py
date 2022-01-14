import os
import json
import shutil

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

ingested_files_path = os.path.join(
    config['output_folder_path'], 'ingestedfiles.txt'
    )
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')  
score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
files_to_copy = [ingested_files_path, model_path, score_path]

prod_deployment_path = config['prod_deployment_path']

#################### Function for deployment #############################
def copy_files_to_deployment() -> None:
    """
    Function to copy latest model, model score and ingest data info to
    the deployment folder.
    """
    
    for file in files_to_copy:
        if os.path.isfile(file):
            shutil.copy(
                src=file,
                dst = os.path.join(prod_deployment_path, file.split('/')[-1])
                )  
    

if __name__ == '__main__':
    copy_files_to_deployment()    

