from typing import Tuple

import pandas as pd
import numpy as np


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to load a csv file in data_path to a Pandas DataFrame.
    The data is then split into inputs and target variable and 
    returned as Numpy arrays.

    Inputs
    ------
    data_path : str
        Filepath of the data to load.
    Returns
    -------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target variable.
    """
    df = pd.read_csv(data_path, index_col='corporation')
    y = df['exited'].values.reshape(-1,1).ravel()
    X = df.drop(columns='exited').values.reshape(-1,3)
    return X, y