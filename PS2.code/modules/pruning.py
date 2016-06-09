from node import Node
from ID3 import *
from operator import xor

# Note, these functions are provided for your reference.  You will not be graded on their behavior,
# so you can implement them as you choose or not implement them at all if you want to use a different
# architecture for pruning.

def reduced_error_pruning(root,training_set,validation_set):
    '''
    take the a node, training set, and validation set and returns the improved node.
    You can implement this as you choose, but the goal is to remove some nodes such that doing so improves validation accuracy.
    NOTE you will probably not need to use the training set for your pruning strategy, but it's passed as an argument in the starter code just in case.
    '''

    if root.label or validation_set == []:
        return root
    else:
        nmode = Node()
        nmode.label = mode(validation_set)

        curracc = validation_accuracy(root,validation_set)
        modeacc = validation_accuracy(nmode,validation_set)

        if modeacc >= curracc:
            return nmode
        elif not root.is_nominal:
            left,right = split_on_numerical(validation_set,root.decision_attribute,root.splitting_value)
            root.children = [reduced_error_pruning(root.children[0],training_set,left),reduced_error_pruning(root.children[1],training_set,right)]
        else:
            subdata = split_on_nominal(validation_set,root.decision_attribute)
            for (key,val) in root.children.iteritems():
                currdata = []
                if key in subdata:
                    currdata = subdata[key]
                root.children[key] = reduced_error_pruning(val,training_set,currdata)

        return root

    pass


def count_splits(tree):
    count = 0

    count += len(tree.children)

    if not tree.is_nominal:
        for child in tree.children:
            count += count_splits(child)
    else:
        for (key,child) in tree.children.iteritems():
            count += count_splits(child)

    return count
# 

def validation_accuracy(tree,validation_set):
    '''
    takes a tree and a validation set and returns the accuracy of the set on the given tree
    '''
    # print "Splits: ",count_splits(tree)

    correct = 0

    for i in range(1,len(validation_set[0])):
        handle_missing_values(validation_set,i)

    for example in validation_set:
        if tree.classify(example) == example[0]:
            correct = correct + 1

    pct = float(correct)/len(validation_set)
    return pct
    pass
