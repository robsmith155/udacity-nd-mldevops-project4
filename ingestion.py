import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


############ Load config.json and get input and output paths ##############
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


################### Function for data ingestion ###########################
def merge_multiple_dataframe() -> None:
    """
    Checks for datasets stored as .csv files, compiles them together
    and then outputs the compiled data as a .csv.
    """
    final_df = pd.DataFrame(
        columns=[ 
            'lastmonth_activity', 
            'lastyear_activity', 
            'number_of_employees', 
            'exited'])

    filenames = os.listdir(input_folder_path)
    csv_files = []
    
    for file in filenames:
        if file.endswith('.csv'):
            csv_files.append(file)
            filepath = os.path.join(input_folder_path, file)
            df = pd.read_csv(filepath, index_col='corporation')
            final_df = final_df.append(df)
        else:
            pass
    
    # Drop dulplicate rows
    final_df.drop_duplicates(inplace=True)

    # Save data
    ingested_filepath = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_filepath, 'w') as f:
        f.write(json.dumps(csv_files))
    
    data_filepath = os.path.join(output_folder_path, 'finaldata.csv')
    final_df.to_csv(data_filepath, index_label='corporation')



if __name__ == '__main__':
    merge_multiple_dataframe()
