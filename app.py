from flask import Flask, request

from diagnostics import (
    model_predictions,
    dataframe_summary, 
    missing_data, 
    execution_time, 
    outdated_packages_list)
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


########################### Prediction Endpoint #############################

@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    Endpoint to return predictions from the trained model.
    """
    data_path = request.args.get('datapath')
    y_preds = model_predictions(data_path=data_path)
    return str(y_preds)

############################ Scoring Endpoint ###############################

@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """
    Endpoint to check the score of the current model on the test dataset
    """
    data_path = request.args.get('datapath')
    score = score_model(data_path)
    return str(score)

####################### Summary Statistics Endpoint #########################

@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    """
    Endpoint to return summary statistics from a selected dataset
    """
    data_path = request.args.get('datapath')
    summary = dataframe_summary(data_path=data_path)     
    return str(summary)

########################### Diagnostics Endpoint ############################

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    Run and resturn results from diagnostic checks including execution time,
    the percentage of missing values from each column in the dataset and 
    a list of the packages that are not using the latest version.
    """
    data_path = request.args.get('datapath')
    col_na_pc = missing_data(data_path=data_path)
    runtimes = execution_time()
    outdated_packages = outdated_packages_list()
    output ={
        'Column missing_values_pc': col_na_pc,
        'Runtimes': runtimes,
        'Outdated packages': outdated_packages
        }
    return str(output)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
