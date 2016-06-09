import math
from node import Node
import sys
import math
from copy import deepcopy


def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================

    '''
    # returned node
    root = Node()

    # if all same values, found a leaf
    same = check_homogenous(data_set)
    if same is not None:
        root.label = same
        return root

    # if run out of depth, return a leaf
    if attribute_metadata == [] or depth == 0:
        root.label = mode(data_set)
        return root

    # get an array with all attribute modes, in proper index
    attmodes = [1]
    for i in range(1,len(attribute_metadata)):
        attmodes.append(handle_missing_values(data_set,i))

    deepsplits = deepcopy(numerical_splits_count)

    best,split = pick_best_attribute(data_set,attribute_metadata,deepsplits)

    # if no more attributes to use, return a leaf
    if best == 0 or best == False:
        root.label = mode(data_set)
        return root

    root.name = attribute_metadata[best]['name']
    root.decision_attribute = best
    root.label = None
    root.value = attmodes[best]

    root.is_nominal = attribute_metadata[best]['is_nominal']

    if root.is_nominal:
        subdata = split_on_nominal(data_set,best)
        # needs a child for every value
        for val in subdata.keys():

            tmp = attribute_metadata[best]
            attribute_metadata[best] = None
            child = ID3(subdata[val],attribute_metadata,deepsplits,depth-1)
            attribute_metadata[best] = tmp
            root.children[val] = child

    else:
        # only gets two children with a splitting value
        root.splitting_value = split
        left,right = split_on_numerical(data_set,best,root.splitting_value)

        deepsplits[best] = deepsplits[best] - 1
        root.children = []
        root.children.append(ID3(left,attribute_metadata,deepsplits,depth-1))
        root.children.append(ID3(right,attribute_metadata,deepsplits,depth-1))

    return root

    pass

def handle_missing_values(data_set,attribute):
    david = []
    # put the attribute values in an array of 1 element arrays
    for example in data_set:
        # don't want the mode to be None.. that wouldn't be good
        if example[attribute] is not None:
            david.append([example[attribute]])

    # that way we can use mode to find the mode
    amode = mode(david)

    # replace the missing values with data set mode
    for example in data_set:
        if example[attribute] is None:
            example[attribute] = amode

    return amode


# passed
def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the output value (index 0) is the same for all examples in the the data_set, if so return that output value, otherwise return None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
     '''
    firstNum = 0
    first = True
    for example in data_set:
        if first:
            firstNum = example[0]
            first = False
        else:
            if example[0] != firstNum:
                return None

    return firstNum
    pass
# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    maxgain = 0.0
    best = 0
    splval = 0.0

    # run through attr
    for i in range(1,len(attribute_metadata)):
        # metadata set to None if attribute has already been split on
        if attribute_metadata[i] is not None:
            if attribute_metadata[i]['is_nominal']:
                gain = gain_ratio_nominal(data_set,i)

                # for the bonus question
                # gain = regular_gain_nominal(data_set,i)
                if gain > maxgain:
                    best = i
                    maxgain = gain
            else:
                # check that we haven't hit the max split count yet
                if numerical_splits_count[i] != 0:
                    gain,split = gain_ratio_numeric(data_set,i)

                    # for the bonus question
                    # gain,split = regular_gain_numeric(data_set,i)
                    if gain > maxgain:
                        best = i
                        maxgain = gain
                        splval = split

    # use False for splitting value for nominal
    if attribute_metadata[best] is not None and attribute_metadata[best]['is_nominal']:
        return best,False
    # no attribute found
    elif maxgain == 0:
        return False,False
    # numeric attribute
    else:
        return best,splval



    pass

# # ======== Test Cases =============================
# numerical_splits_count = [20,20]
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''
    sven = {}

    for example in data_set:
        val = example[0]
        if val not in sven:
            sven[val] = 1
        else:
            sven[val] += 1

    return max(sven,key=sven.get)
# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''
    nums = {}
    entropy = 0

    for example in data_set:
        val = example[0]
        if val not in nums:
            nums[val] = 1
        else:
            nums[val] += 1
    total = 0
    for val in nums.values():
        total += val
    for key in nums.keys():
        prob = float(nums[key])/total
        entropy -= (prob * math.log(prob,2))

    return entropy


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    nums = {}
    # sum of entropies for each value
    subent = 0.0
    # sum of intrinsic values for each value
    intrval = 0.0

    # get the counts of each value of attribute
    for example in data_set:
        val = example[attribute]
        if val not in nums:
            nums[val] = 1
        else:
            nums[val] += 1

    total = len(data_set)

    # get the subdata for each value
    subs = split_on_nominal(data_set,attribute)

    # calculate the IV and subentropy
    for val in nums.keys():
        prob = float(nums[val])/total
        subdata = subs[val]
        subent += prob*entropy(subdata)
        intrval += prob*math.log(prob,2)

    # IG
    gain = entropy(data_set) - subent
    # handle divide by 0 to get the IGR
    if intrval == 0:
        ratio = 0
    else:
        ratio = gain/(-1*intrval)
    return ratio

    pass
# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps=1):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    thresholds = []

    # make there fewer steps to decrease training time
    steps = max(1,(int)(len(data_set)/3))

    # get numeric thresholds
    for i in range(0,len(data_set)):
        if i % steps == 0:
            thresholds.append(data_set[i][attribute])

    thresh = 0
    maxgain = 0.0

    total = float(len(data_set))

    # calculate IGR for the two data sets per threshold
    for val in thresholds:
        intrval = 0.0
        subent = 0.0

        (low,high) = split_on_numerical(data_set,attribute,val)
        lprob = len(low)/total
        hprob = 1 - lprob


        subent = lprob*entropy(low) + hprob*entropy(high)

        if lprob == 0 or hprob == 0:
            ratio = 0
        else:
            intrval = lprob*math.log(lprob,2) + hprob*math.log(hprob,2)
            ratio = (entropy(data_set) - subent)/(-1*intrval)

        if ratio > maxgain:
            thresh = val
            maxgain = ratio

    return (maxgain,thresh)



    # Your code here
    pass
# ======== Test case =============================
# data_set,attr,step = [[0,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)


def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    jack = {}

    for example in data_set:
        val = example[attribute]
        if val not in jack:
            jack[val] = [example]
        else:
            jack[val].append(example)

    return jack
    pass
# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    bigyoss = []
    jr = []

    for example in data_set:
        # attribute < splitting value vs attribute >= splitting value
        if example[attribute] < splitting_value:
            bigyoss.append(example)
        else:
            jr.append(example)

    return (bigyoss,jr)
    pass
# ======== Test case =============================
# d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
# split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
# d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
# split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])

# bonus question 11 functions
def regular_gain_nominal(data_set,attribute):
    nums = {}
    # sum of entropies for each value
    subent = 0.0
    # sum of intrinsic values for each value
    intrval = 0.0

    # get the counts of each value of attribute
    for example in data_set:
        val = example[attribute]
        if val not in nums:
            nums[val] = 1
        else:
            nums[val] += 1

    total = len(data_set)

    # get the subdata for each value
    subs = split_on_nominal(data_set,attribute)

    # calculate the IV and subentropy
    for val in nums.keys():
        prob = float(nums[val])/total
        subdata = subs[val]
        subent += prob*entropy(subdata)
        intrval += prob*math.log(prob,2)

    # IG
    gain = entropy(data_set) - subent

    return gain


    pass

def regular_gain_numeric(data_set,attribute,steps=1):
    thresholds = []

    # make there fewer steps to decrease training time
    steps = max(1,(int)(len(data_set)/3))

    # get numeric thresholds
    for i in range(0,len(data_set)):
        if i % steps == 0:
            thresholds.append(data_set[i][attribute])

    thresh = 0
    maxgain = 0.0

    total = float(len(data_set))

    # calculate IGR for the two data sets per threshold
    for val in thresholds:
        intrval = 0.0
        subent = 0.0

        (low,high) = split_on_numerical(data_set,attribute,val)
        lprob = len(low)/total
        hprob = 1 - lprob


        subent = lprob*entropy(low) + hprob*entropy(high)

        gain = (entropy(data_set) - subent)

        if gain > maxgain:
            thresh = val
            maxgain = gain

    return (maxgain,thresh)


    pass





