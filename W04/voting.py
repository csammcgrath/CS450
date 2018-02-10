import pandas as pd

def get_voting():
    headers = [
        'party',                #1
        'infant',               #2
        'water',                #3
        'budget-resolution',    #4
        'physician-freeze',     #5
        'salvador-aid',         #6
        'religion-school',      #7
        'anti-satellite',       #8
        'nicaraguan-aid',       #9
        'mx-missile',           #10
        'immigration',          #11
        'synfuels-corporation', #12
        'education',            #13
        'superfund',            #14
        'crime',                #15
        'duty-free-exports',    #16
        'export-south-africa'   #17
    ]

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",
                        delimiter=",", header=None, name=headers)

    return df[headers[1:17]], df[headers[0:1]], headers