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
import Node

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
clean_df.to_csv("clean_data1.csv", sep=',')
#print(clean_df.iloc[:,19:25])
tree = Node.Node(clean_df,1)
#tree.print_tree(0)
#print("original entropy: ",Node.entropy(clean_df,1))
#Node.info_gain(clean_df, 1)

# print(new_clean_df.loc[:,1].var())
# print(clean_df.loc[clean_df[1] == 1].shape[0])
# print(clean_df)
# print(new_clean_df)
noisy_df = Load_panda("./Data/noisydata_students.mat",'x')
noisy_labels_df = Load_panda("./Data/noisydata_students.mat",'y')
noisy_df = noisy_df.assign(label = noisy_labels_df)
