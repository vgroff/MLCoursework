# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Function to read in the rar inputs
from numpy import *
from scipy.io import *
import pandas as pda
import numpy as np
import Model
import Tree
import datetime
import os


def Load_panda(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    output = pda.DataFrame(array)

    return output

def getResults():
    dt = datetime.datetime.now().strftime("%H:%M_%d-%m")
    folder = "results_" + dt
    os.makedirs(folder)
    resultsFile = open(folder + "/" + "results.txt", "w")
    dataSets = [ ["./Data/cleandata_students.mat", "Clean data", True],
             ["./Data/noisydata_students.mat", "Noisy data", False] ]
    for dataSet in dataSets:
        df = Load_panda(dataSet[0],'x')
        lean_labels_df = Load_panda(dataSet[0],'y')
        df = df.assign(label = clean_labels_df)
        validation_df, test_df = Model.split(0.8, df)
        unpruned_conf_matrix, pruned_conf_matrix = Model.crossValidate(validation_df, 10, folder)
        unpruned_results = Model.performanceMetricsDF(unpruned_conf_matrix, False, dataSet[2], folder)
        pruned_results = Model.performanceMetricsDF(pruned_conf_matrix, True, dataSet[2], folder)
        resultsFile.write(dataSet[1] + "\n")
        resultsFile.write("Unpruned Results\n")
        resultsFile.write(unpruned_conf_matrix.__str__())
        resultsFile.write("\n")
        resultsFile.write(unpruned_results.__str__())
        resultsFile.write("\n\nPruned Results\n")
        resultsFile.write(pruned_conf_matrix.__str__())
        resultsFile.write("\n")
        resultsFile.write(pruned_results.__str__())
        resultsFile.write("\n\n")
    resultsFile.close()



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

getResults()


#validation_df, test_df = Model.split(0.8, clean_df)
#unpruned_conf_matrix, pruned_conf_matrix = Model.crossValidate(validation_df, 10)
#unpruned_results = Model.performanceMetrics(unpruned_conf_matrix)
#pruned_results = Model.performanceMetrics(pruned_conf_matrix)
#print(unpruned_results)
#print(pruned_results)

# test = clean_df.iloc[0:900,:]
# val = clean_df.iloc[901:1003,:]
# model = Model.Model(test)
# classified = []
# real = []
# count = 0
# #print(test)
# #print(val)
# for i in range(0,val.shape[0]):
#     test_row = val.iloc[i,:]
#     #test_row = test.loc(i)
#     real.append(test_row.loc['label'])
#     classified.append(model.classify(test_row))
#     print( real[-1], classified[-1], model.rawClassify(test_row))
#     if (real[-1] == classified[-1]):
#         count += 1
#     else:
#         print("different")
# print(count)

# print(new_clean_df.loc[:,1].var())
# print(clean_df.loc[clean_df[1] == 1].shape[0])
# print(clean_df)
# print(new_clean_df)
noisy_df = Load_panda("./Data/noisydata_students.mat",'x')
noisy_labels_df = Load_panda("./Data/noisydata_students.mat",'y')
noisy_df = noisy_df.assign(label = noisy_labels_df)
