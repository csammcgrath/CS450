import pandas as pd
import numpy as np

def get_shuttle_dataset():
    headers = [
        'class',
        'stability',
        'error',
        'sign',
        'wind',
        'magnitude',
        'visibility'
    ]

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/shuttle-landing-control/shuttle-landing-control.data', \
                     header=None, names=headers)
    
    df.replace('*', 0, inplace = True)

    return df.as_matrix(columns=headers[1:]), df.as_matrix(column=headers[:1])