from typing import Tuple

import pandas as pd
import numpy as np


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path, index_col='corporation')
    y = df['exited'].values.reshape(-1,1).ravel()
    X = df.drop(columns='exited').values.reshape(-1,3)
    return X, y