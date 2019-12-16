# Anant Ahuja
# axa9357
#--------#---------#---------#---------#--------#--------#---------#---------#---------#--------#
import sys
import csv
import numpy as np
from random import shuffle
#--------#---------#---------#---------#--------#--------#---------#---------#---------#--------#
def main():

   # Parsing the validating dataset and loading it
    dataset = open(sys.argv[1])
    dataset_reader = csv.reader(dataset)
    data = list(dataset_reader)
    dataset.close()

    # Converting the labels from strings to integer values 1,2 and 3
    for row in data:
        if row[4] == 'Iris-setosa':
            row[4] = 1
        elif row[4] == 'Iris-versicolor':
            row[4] = 2
        else:
            row[4] = 3

    # Creating 3 new lists for each of the labels. This is to ensure equal distribution of labels in all 5 segments of the data.
    # Note: These lists can be shuffled. I have commented out the shuffle part of the code so the TA gets the same accuracy and Beta values as the ones included in the report.
    setosa_list = data[0:50]
    versicolor_list = data[50:100]
    virginica_list = data[100:150]

    # shuffle(setosa_list)
    # shuffle(versicolor_list)
    # shuffle(virginica_list)

    # Performing 5-fold cross validation. So splitting data into 5 different segments with 10 datapoints from each label.
    segments = {}
    segments['0'] = setosa_list[0:10] + versicolor_list[0:10] + virginica_list[0:10]
    segments['1'] = setosa_list[10:20] + versicolor_list[10:20] + virginica_list[10:20]
    segments['2'] = setosa_list[20:30] + versicolor_list[20:30] + virginica_list[20:30]
    segments['3'] = setosa_list[30:40] + versicolor_list[30:40] + virginica_list[30:40]
    segments['4'] = setosa_list[40:50] + versicolor_list[40:50] + virginica_list[40:50]

    Beta_values = []

    # 5 iterations
    for i in range(5):
        testing_set = []
        training_set = []

        # Partitioning the training set and the testing set
        for k in segments.keys():
            if k == str(i):
                testing_set = testing_set + segments[k]
            else:
                training_set = training_set + segments[k]

        # Creating all the matrices required for the linear regression formula
        training_matrix = np.array(training_set, dtype=np.float64)
        A = training_matrix[ : , 0 : 4].copy()
        AT1 = A.copy().T
        AT2 = A.copy().T
        Y = training_matrix[ : , -1].copy()

        # Beta value for all 4 features
        Beta = np.linalg.inv(AT1 @ A) @ AT2 @ Y
        Beta_values.append(Beta)

        # Creating all the matrices required for testing our Beta on the testing set
        testing_matrix = np.array(testing_set, dtype=np.float64)
        B = testing_matrix[ : , 0 : 4].copy()
        BT1 = B.copy().T
        BT2 = B.copy().T

        # Testing the Beta value on the test set
        predicted_values = []
        for row in B:
            predicted_values.append(Beta @ row)

        # Calculating the accuracy of results on our test set
        correct_count = 0
        for datapoint, label in zip(testing_set, predicted_values):
            if round(label) == datapoint[4]:
                correct_count += 1

        accuracy = (correct_count / 30) * 100
        print("Beta = {}, Accuracy = {} for this testing set.".format(Beta, accuracy))

    # Taking an average of all the Beta values and testing it for the entire set.
    temp = np.array(Beta_values)
    Average_Beta = np.mean(temp, axis=0)

    # Using this average Beta value and testing over the entire data set.
    dataset_matrix = np.array(data, dtype=np.float64)
    C = dataset_matrix[ : , 0 : 4].copy()
    CT1 = C.copy().T
    CT2 = C.copy().T

    predicted_values_final = []
    for row in C:
        predicted_values_final.append(Average_Beta @ row)

    correct_count_final = 0
    for datapoint, label in zip(data, predicted_values_final):
        if round(label) == datapoint[4]:
            correct_count_final += 1

    # Printing Average Beta valie and final accuracy
    accuracy_final = (correct_count_final / 150) * 100
    print("Average Beta = {}, Total Accuracy = {} for the entire dataset.".format(Average_Beta, accuracy_final))

if __name__ == '__main__':
    main()