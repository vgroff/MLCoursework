from numpy import *
from scipy.io import *
import pandas as pda
import numpy as np
import Model
import Tree

def Load_panda(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    output = pda.DataFrame(array)

    return output

clean_df = Load_panda("./Data/cleandata_students.mat",'x')
clean_labels_df = Load_panda("./Data/cleandata_students.mat",'y')
clean_df = clean_df.assign(label = clean_labels_df)

training_df, test_df = Model.split(0.8, clean_df)

unpruned_conf_matrix, pruned_conf_matrix = Model.crossValidate(validation_df, 10)
unpruned_results = Model.performanceMetrics(unpruned_conf_matrix)
pruned_results = Model.performanceMetrics(pruned_conf_matrix)

print("Unpruned Results")
print(unpruned_results)
print("Pruned Results")
print(pruned_results)

noisy_df = Load_panda("./Data/noisydata_students.mat",'x')
noisy_labels_df = Load_panda("./Data/noisydata_students.mat",'y')
noisy_df = noisy_df.assign(label = noisy_labels_df)
