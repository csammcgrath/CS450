import pandas as pd
import numpy as np
from decision_tree import build_tree
from tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn import datasets as sk_dataset
from loan import get_loans
from voting import get_voting

def main():
    try:
        print("ID3 Algorithm - CS 450 BYU-Idaho")
        print("Which dataset would you like to use?")
        print("1 - loan (simple)")
        print("2 - voting (hard)")

        id3_dataset = int(input("> "))

        if (id3_dataset < 1 or id3_dataset > 2):
            raise Exception('invalid dataset selection')
        elif (id3_dataset == 1 or id3_dataset == 2):
            execute_algorithm(id3_dataset)
        else:
            raise Exception("unknown error.")
    except (ValueError) as err:
        print('ERROR: {}'.format(err))

# execute_algorithm()
#
#This function goes through and begins the execution of the ID3 tree.
def execute_algorithm(dataset):
    #we all know that this whole shell is designed just for the Decision Tree
    classifier = DecisionTree()

    #determine which dataset to retrieve
    if (dataset == 1):
        data, targets, headers = get_loans()
    elif (dataset == 2):
        data, targets, headers = get_voting()
    count = 0

    #split dataset into random parts
    train_data, test_data, train_target, test_target = split_data(data, targets)

    #reset the indexes so the dataframe can be properly parsed.
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    train_target.reset_index(inplace=True, drop=True)
    test_target.reset_index(inplace=True, drop=True)

    #build the tree!
    model = classifier.fit(train_data, train_target, headers)

    #prompt the user if he/she wants to display the tree
    print_id3(model)

    #target_predicted is an array of predictions that is received by the predict
    target_predicted = model.predict(test_data)

    #this allows us to know which column is the target
    test_target = test_target[headers[-1]]

    #loop through the target_predicted and count up the correct predictions
    for index in range(len(target_predicted)):
        #increment counter for every match from
        #target_predicted and test_target
        if target_predicted[index] == test_target[index]:
            count += 1

    accuracy = get_accuracy(count, len(test_data))

    #report to the user
    print("Accuracy: {:.2f}%".format(accuracy))

#print_id3
#
#this function prompts the user if he/she wants to see the tree
def print_id3(tree):
    try:
        print("Would you like to print the ID3 Tree? ")
        print("0 - No")
        print("1 - Yes")

        printID3 = int(input("> "))

        if (printID3 < 0 or printID3  > 1):
            raise Exception('invalid print option.')
        elif (printID3 == 1):
            print(tree)
    except (ValueError) as err:
        print("ERROR: {}".format(err))

# get_accuracy()
#
# This function calculates and returns the accuracy
def get_accuracy(count, length):
    return (count / length) * 100

# split_data
#
# This function was designed to help make the execute_algorithm look cleaner.
def split_data(data, target):
    return train_test_split(data, target, test_size=.3)

if __name__ == "__main__":
    main()