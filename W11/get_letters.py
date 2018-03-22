import pandas as pd
import numpy as np

def get_letters_dataset():
    headers = [
        'class',
        'x-box',
        'y-box',
        'width',
        'onpix',
        'x-bar',
        'y-bar',
        'x2bar',
        'y2bar',
        'xybar',
        'x2ybar',
        'xy2bar',
        'x-ege',
        'xegvy',
        'y-ege',
        'yegvx'
    ]

    replaceObj = {
        'class': {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 7,
            'I': 8,
            'J': 9,
            'K': 10,
            'L': 11,
            'M': 12,
            'N': 13,
            'O': 14,
            'P': 15,
            'Q': 16,
            'R': 17,
            'S': 18,
            'T': 19,
            'U': 20,
            'V': 21,
            'W': 22,
            'X': 23,
            'Y': 24,
            'Z': 25,
        }
    }

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data', \
                    header=None, names=headers)

    df.replace(replaceObj, inplace=True)

    return df.as_matrix(columns=headers[1:]), df.as_matrix(columns=headers[:1])