from tree import DecisionTree
from decision_tree import build_tree
from sklearn.model_selection import train_test_split
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

def execute_algorithm(dataset):
    #we all know that this whole shell is designed just for the Decision Tree
    classifier = DecisionTree()

    if (dataset == 1):
        data, targets, headers = get_loans()
    elif (dataset == 2):
        data, targets, headers = get_voting()
    count = 0

    train_data, test_data, train_target, test_target = split_data(data, targets)

    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    train_target.reset_index(inplace=True, drop=True)
    test_target.reset_index(inplace=True, drop=True)

    model = classifier.fit(train_data, train_target, headers)

    target_predicted = model.predict(test_data)

    test_target = test_target[headers[-1]]

    for index in range(len(target_predicted)):
        #increment counter for every match from
        #target_predicted and test_target
        if target_predicted[index] == test_target[index]:
            count += 1

    accuracy = get_accuracy(count, len(test_data))

    print("Accuracy: {:.2f}%".format(accuracy))

# get_accuracy()
#
# This function calculates and returns the accuracy
def get_accuracy(count, length):
    return (count / length) * 100

def split_data(data, target):
    return train_test_split(data, target, test_size=.3)

if __name__ == "__main__":
    main()