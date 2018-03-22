from sklearn import ensemble
from sklearn import preprocessing
from sklearn import tree
from sklearn import neural_network
from sklearn import neighbors
from sklearn import naive_bayes

from get_shuttle import get_shuttle_dataset 
from get_letters import get_letters_dataset
from get_iris import get_iris_dataset

def main():
    try:
        print("Ensemble Algorithm BYU-Idaho")
        print("---------------------------------")
        print("Please select a dataset (all can be found in UCI repository")
        print("1 - Shuttle Landing Control")
        print("2 - Iris")
        print("3 - Letter Recognition")

        dataset = int(input("> "))

        if (dataset < 1 or dataset > 3):
            raise Exception("invalid dataset choice")
    except ValueError as err:
        print("ERROR: {}".format(err))



def test(self, dataset):
    data = None
    targets = None

    if dataset == 1:
        data, targets = get_shuttle_dataset()
    elif dataset == 2:
        data, targets = get_iris_dataset()
    else:
        data, targets = get_letters_dataset()

if __name__ == "main()":
    main()
