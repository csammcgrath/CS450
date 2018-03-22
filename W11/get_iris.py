import pandas as pd
import numpy as np

def get_iris_dataset():
    headers = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'class'
    ]

    replaceObj = {
        'class': {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
    }

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', \
                     header=None, names=headers)

    df.replace(replaceObj, inplace=True)

    return df.as_matrix(columns=headers[:4]), df.as_matrix(columns=headers[:-1])