# -*- coding: utf-8 -*-

# Function to read in the rar inputs
from numpy import *
from scipy.io import *
import pandas as pda
import numpy as np
import Model
import Tree
import datetime
import os
import pickle

# Load a .mat files as a panda data frame
def Load_panda(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    output = pda.DataFrame(array)

    return output

# Produce performance metris and confusion matrices for both datasets
def getResults():
    dt = datetime.datetime.now().strftime("%H:%M_%d-%m")
    folder = "results_" + dt
    os.makedirs(folder)
    resultsFile = open(folder + "/" + "results.txt", "w")
    dataSets = [ ["./Data/cleandata_students.mat", "Clean data", True],
                 ["./Data/noisydata_students.mat", "Noisy data", False] ]
    for dataSet in dataSets:
        df = Load_panda(dataSet[0],'x')
        labels_df = Load_panda(dataSet[0],'y')
        df = df.assign(label = labels_df)
        df = df.sample(frac=1).reset_index(drop=True)
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

# Take a .mat filename, load the data, classify the data using a model loaded from file
def classifyEmotions(filename):
    df = Load_panda(filename,'x')
    labels_df = Load_panda(filename,'y')
    df = df.assign(label = labels_df)
    df = df.sample(frac=1).reset_index(drop=True)
    f = open("./Final_Model/model", "rb")
    model = pickle.load(f)
    f.close()
    return model.test_model(df)

# Produce a model from .mat data and save it to file using pickle
def saveTrees(dataFilename):
    df = Load_panda(dataFilename,'x')
    labels_df = Load_panda(dataFilename,'y')
    df = df.assign(label = labels_df)
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, prune_df = Model.split(0.8, df)

    model = Model.Model(train_df)
    model.prune(prune_df)

    model.print_to_file("./Final_Model", True)
    model.print_to_file("./Final_Model", False)

    f = open("./Final_Model/model", "wb")
    pickle.dump(model, f)
    f.close()
