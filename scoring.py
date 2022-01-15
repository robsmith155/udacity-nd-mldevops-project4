import pickle
import os
from sklearn import metrics
import json

from utils import load_data

################# Load config.json and get path variables ###################
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl') 
score_path = os.path.join(config['output_model_path'], 'latestscore.txt')

################# Function for model scoring ################################
def score_model(data_path: str):
    """
    Function to score the trained model using the F1 metric on teh test
    dataset. Score is output to a file named 'latestscore.txt' in the 
    trained model folder.
    """
    X, y = load_data(data_path)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    score = metrics.f1_score(y_true=y, y_pred=y_pred)
    
    with open(score_path, 'w') as f:
        f.write(str(score))
    
    return score


if __name__ == '__main__':
    score_model(test_data_path)