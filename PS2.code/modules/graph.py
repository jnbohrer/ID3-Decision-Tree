from random import shuffle
from ID3 import *
from operator import xor
from parse import parse
import matplotlib.pyplot as plt
import os.path
from pruning import *

# NOTE: these functions are just for your reference, you will NOT be graded on their output
# so you can feel free to implement them as you choose, or not implement them at all if you want
# to use an entirely different method for graphing

def get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, pct):
    '''
    get_graph_accuracy_partial - Given a training set, attribute metadata, validation set, numerical splits count, and percentage,
    this function will return the validation accuracy of a specified (percentage) portion of the trainging setself.
    '''

    # start = random.randint(0,len(train_set) - pct*len(train_set))
    # partdata = []
    # for i in range(start,start+pct*len(train_set)):
    #     partdata.append(train_set[i])

    partdata = []
    shuffle(train_set)
    for i in range(0,(int)(pct*len(train_set))):
        partdata.append(train_set[i])

    tree = ID3(partdata,attribute_metadata,numerical_splits_count,depth=10)

    # change from tree to prunetree if pruning is turned on
    prunetree = reduced_error_pruning(tree,train_set,validate_set)

    val = validation_accuracy(prunetree,validate_set)
    print val
    return val

    pass

def get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, iterations, pcts):
    '''
    Given a training set, attribute metadata, validation set, numerical splits count, iterations, and percentages,
    this function will return an array of the averaged graph accuracy partials based off the number of iterations.
    '''
    avgpartials = []

    for i in range(0,len(pcts)):
        print pcts[i]
        avgpartials.append(0)
        for x in range(0,iterations):
            avgpartials[i] += get_graph_accuracy_partial(train_set,attribute_metadata,validate_set,numerical_splits_count,pcts[i])

    for i in range(0,len(avgpartials)):
        avgpartials[i] = avgpartials[i] / iterations;

    return avgpartials

    pass

# get_graph will plot the points of the results from get_graph_data and return a graph
def get_graph(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, lower, upper, increment):
    '''
    get_graph - Given a training set, attribute metadata, validation set, numerical splits count, depth, iterations, lower(range),
    upper(range), and increment, this function will graph the results from get_graph_data in reference to the drange
    percentages of the data.
    '''
    pcts = []
    i=0.1
    while (i <= 1):
        pcts.append(i)
        i = i + .1

    avgs = get_graph_data(train_set,attribute_metadata,validate_set,numerical_splits_count,iterations,pcts)

    plt.plot(pcts,avgs)
    plt.show()

    pass