from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import json
import os

from diagnostics import model_predictions
from utils import load_data

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
conf_matrix_path = os.path.join(config['output_model_path'], 'confusionmatrix.png') 


##############Function for reporting
def score_model() -> None:
    """
    Outputs a confusion matrix for the test dataset.
    """
    y_preds = model_predictions(data_path=test_data_path)
    _, y_true = load_data(data_path=test_data_path)

    cm = confusion_matrix(y_true, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(conf_matrix_path)


if __name__ == '__main__':
    score_model()
