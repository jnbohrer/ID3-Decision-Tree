import os.path
from operator import xor
from parse import *
from ID3 import *
from copy import deepcopy

# DOCUMENTATION
# ========================================
# this function outputs predictions for a given data set.
# NOTE this function is provided only for reference.
# You will not be graded on the details of this function, so you can change the interface if 
# you choose, or not complete this function at all if you want to use a different method for
# generating predictions.

def convertCSV(row):
    data = [None]*len(row)
    for i in range(len(row)):
        if (row[i] == '?'):
            data[i] = None
        else:
            data[i] = float(row[i])

    return data


def create_predictions(tree, predict):
    '''
    Given a tree and a url to a data_set. Create a csv with a prediction for each result
    using the classify method in node class.
	'''

    # our own parse function
    csv_test = csv.reader(open(predict))
    csv_test.next()
    output = []

    for row in csv_test:
        output.append(convertCSV(row))

    # part we were missing - rotating the winning value to the front
    classdata = []
    for point in output:
        d = deepcopy(point)
        d = collections.deque(d)
        d.rotate(1)
        classdata.append(list(d))

    for i in range(1,len(output[0])):
        handle_missing_values(classdata,i)


    # start the classification and writing
    last = len(output[0]) - 1
    for i in range(0,len(output)):
        output[i][last] = tree.classify(classdata[i])

    w = csv.writer(open('data/PS2.csv', 'w'))
    w.writerows(output)

    # for point in output:
    #     point[len(point)-1] = tree.classify(point)
    # w = csv.writer(open('data/PS2.csv', 'w'))
    # w.writerows(output)




