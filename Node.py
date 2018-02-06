# -*- coding: utf-8 -*-*~
import pandas as pd
import numpy as np
import random

class Node:
    def __init__(self, df, binary_target):
        self.children = []
        self.variable = None
        self.input_prob = None
        # boolean classifier based on whether it equals the binary
        self.classification = None

        # track the number of instances that do not/match the binary_target
        self.true_num = None
        self.false_num = None
        self.pruned = False
        self.prune_attempted = False

        # we will define a classification [p,1-p] as the probability of being in class binary_target
        # this is based on the incoming distribution of the node in the training data.
        # check if the target variable entropy is 0 or no only the class column remains
        incoming_entropy = entropy(df,binary_target)
        if(incoming_entropy==0 or len(df.columns) < 2 or df.shape[0]==0):
            self.input_prob = probability(df,binary_target)
            self.classification = return_class(self.input_prob)
            return

        col_str,outgoing_entropy = info_gain(df,binary_target)
        self.variable = col_str

        if(outgoing_entropy >= incoming_entropy):
            self.variable = None
            self.input_prob = probability(df,binary_target)
            self.classification = return_class(self.input_prob)
            return

        length = df.shape[0]
        if (length==0):
            print("legth zero error")

        self.true_num = df.loc[df["label"] == binary_target].shape[0]
        self.false_num = length - self.true_num

        left_df = df.loc[df[col_str] == 0]
        right_df = df.loc[df[col_str] == 1]

        del left_df[col_str]
        del right_df[col_str]


        self.children.append(Node(left_df,binary_target))
        self.children.append(Node(right_df,binary_target))


    def node_classify(self,test_df):
        # if(self.pruned):
        #     print("pruned")

        # return a classification if you have hit a leaf node
        if(self.classification!=None):
            if(self.classification==True):
                # print("classification: ", self.classification, " input_prob: ", self.input_prob)
                return self.classification,self.input_prob[0]
            # print("classification: ", self.classification, " input_prob: ", self.input_prob)
            return self.classification,self.input_prob[1]

        # otherwise, sort into left/right based on the value of the variable column
        if(test_df[self.variable]==0):
            return self.children[0].node_classify(test_df)
        else:
            return self.children[1].node_classify(test_df)

    def prunning_change(self):
        self.input_prob = [float(self.true_num)/float(self.true_num+self.false_num),
                           float(self.false_num)/float(self.true_num+self.false_num)]
        self.classification = return_class(self.input_prob)
        self.pruned = True
        self.variable = None
        self.children = None
        self.prune_attempted = True
        print("Prunning changes classification: ",self.classification, " input_prob: ", self.input_prob)


def return_class(prob_array):
    if(prob_array[0]>prob_array[1]):
        return True
    elif(prob_array[0]<prob_array[1]):
        return False
    else:
        # return a random yes/no if the probabilities are equal
        return bool(random.getrandbits(1))

# This is the distribution of the binary_target for each Node, produced from the training set
def probability(df,binary_target):
    dist = [0,0]

    length = df.shape[0]

    if (length==0):
        return dist

    yes = df.loc[df["label"] == binary_target].shape[0]
    no = length - yes

    return [yes/(yes+no),no/(yes+no)]


def info_gain(df, binary_target):
    # print("incoming data frame: ",df)
    min_col = None
    min_entropy = None
    # print("incoming entropy: ",entropy(df,binary_target))
    # generate column list
    column_list = list(df)
    # for all columns in the dataframe except for the class label column
    for i in range(len(column_list)-1):

        one_df = df.loc[df[column_list[i]] == 1]
        zero_df = df.loc[df[column_list[i]] == 0]

        one_count = one_df.shape[0]
        zero_count = zero_df.shape[0]

        # print("info gain: column(",column_list[i],"), one_count(",one_count,"), zero_count(",zero_count,")")

        one_entropy = entropy(one_df,binary_target)
        zero_entropy = entropy(zero_df,binary_target)
        # ws is the remaining entropy in the system after filtering by the column
        ws = (one_count*one_entropy + zero_count*zero_entropy)/(one_count+zero_count)
        # print("ws", ws, " i",i)
        if((min_entropy == None) or (ws < min_entropy)):
            min_entropy = ws
            min_col = column_list[i]

    # print("min_col: ",min_col,"min entropy: ",min_entropy)
    return min_col,min_entropy

def entropy(df, binary_target):
    length = df.shape[0]

    if (length==0):
        return 0

    yes = df.loc[df["label"] == binary_target].shape[0]
    no = length - yes
    if(yes!=0):
        yes_cont = -((yes/length) * np.log2(yes/length))
    else:
        yes_cont = 0

    if(no!=0):
        no_cont = -((no/length) * np.log2(no/length))
    else:
        no_cont = 0

    # if(yes_cont+ no_cont ==1):
    #   print(df)
    return  yes_cont + no_cont

def uniform_df(df,binary_target):
    if(entropy(df,binary_target)==0):
        return True
    return False
