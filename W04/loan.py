import pandas as pd

def get_loans():
    headers = [
        "credit score",
        "income",
        "collateral",
        "should_loan"
    ]

    df = pd.read_csv("loan.csv", delimiter=",", header = None, names = headers)

    return df[headers[0:3]], df[headers[3:4]]