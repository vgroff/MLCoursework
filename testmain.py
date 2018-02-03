# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Function to read in the rar inputs
from numpy import *
from scipy.io import *
from graphviz import *
import pandas as pda
import numpy as np
import Model
import Tree


def Load_panda(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    output = pda.DataFrame(array)

    return output


# read in the files and labels from the .mat into Panda arrays

clean_df = Load_panda("./Data/cleandata_students.mat",'x')
# print(clean_df)
clean_labels_df = Load_panda("./Data/cleandata_students.mat",'y')
#clean_df.append(clean_labels_df)
clean_df = clean_df.assign(label = clean_labels_df)
# print(clean_df)
new_clean_df = clean_df.loc[clean_df[1] == 1]
#clean_df.to_csv("clean_data1.csv", sep=',')
#print(clean_df.iloc[:,19:25])
#tree_array = []

#tree = Tree.Tree(clean_df,5)
#Tree.print_tree(tree.root_node)

#for i in range(0,6):
#    tree = Tree.Tree(clean_df,i)
#    tree_array.append(tree)
#    
##
test_row = clean_df.iloc[13,:]
#classification = []
#for i in range(0,6):
#    classification.append(tree_array[i].classify(test_row))
#
#print(classification)
#test_col = clean_df.loc[:,"label"]
#Tree.print_tree(tree.root_node)
#print("Test Row: ",test_row)
#
#
#print(Tree.confusion_matrix(test_col,test_col))

#print(tree.classify(test_row))

#test = Tree.test_sets(clean_df)

#print_test = tree.root_nodes[0]

#test = test_sets(clean_df)
#print(test)


#print("test size: ",len(test))
#print(test[0])
#print(test[9])
#print(tree.print_tree())
#print("original entropy: ",Node.entropy(clean_df,1))
#Node.info_gain(clean_df, 1)

#model = Model.Model(clean_df)
#print("testing", test_row)
#print(model.classify(test_row))

Model.crossValidate(clean_df)

# print(new_clean_df.loc[:,1].var())
# print(clean_df.loc[clean_df[1] == 1].shape[0])
# print(clean_df)
# print(new_clean_df)
noisy_df = Load_panda("./Data/noisydata_students.mat",'x')
noisy_labels_df = Load_panda("./Data/noisydata_students.mat",'y')
noisy_df = noisy_df.assign(label = noisy_labels_df)
