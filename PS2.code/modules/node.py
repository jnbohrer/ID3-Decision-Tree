# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 nodes if numeric, and a dictionary (key=attribute value, value=node) if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is None if there is a decision attribute, and is the output label (0 or 1 for
#	the homework data set) if there are no other attributes
#       to split on or the data is homogenous
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None

    # passed
    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        if self.label is not None:
            return self.label
        else:
            if not self.is_nominal:
                if instance[self.decision_attribute] < self.splitting_value:
                    return self.children[0].classify(instance)
                else:
                    return self.children[1].classify(instance)
            else:
                if instance[self.decision_attribute] in self.children:
                    return self.children[instance[self.decision_attribute]].classify(instance)
                else:
                    return self.children[self.value].classify(instance)
	pass

    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        IMPLEMENTING THIS FUNCTION IS OPTIONAL
        '''
        # Your code here
        pass


    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''

        return self.dnf_helper("")

        pass

    def dnf_helper(self,formula):
        if self.label is not None:
            if self.label == 1:
                return "(" + formula[3:] + ")"
            else:
                return ""

        elif self.is_nominal:
            help = ""
            for (key,val) in self.children.iteritems():
                thing = val.dnf_helper(formula + " ^ " + self.name + "=" + str(key))
                if thing is not "":
                    if help is not "":
                        help = help + ' v '
                    help = help + thing

            return help
        else:
            help = ""
            for key in range(0,len(self.children)):
                child = self.children[key]
                if key == 0:
                    thing  = child.dnf_helper(formula + " ^ " + self.name + "<" + str(self.splitting_value))
                else:
                    thing = child.dnf_helper(formula + " ^ " + self.name + ">=" + str(self.splitting_value))
                if thing is not "":
                    if help is not "":
                        help = help + ' v '
                    help = help + thing

            return help
