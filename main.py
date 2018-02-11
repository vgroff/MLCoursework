from numpy import *
from scipy.io import *
import pandas as pd
import numpy as np
import Model
import Tree
import datetime
import os
import pickle

def LoadAsDF(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    df = pd.DataFrame(array)
    return df

def EvaluateModel(folder, df, is_clean):
    training_df, test_df = Model.split(0.8, df)

    models, unpruned_conf_matrix_cv, pruned_conf_matrix_cv = Model.crossValidate(training_df, 3, folder, is_clean)
    unpruned_results_cv = Model.performanceMetrics(unpruned_conf_matrix_cv)
    pruned_results_cv = Model.performanceMetrics(pruned_conf_matrix_cv)

    Model.resultsToCSV(unpruned_results_cv, False, is_clean, True, folder)
    Model.confusionMatrixToCSV(unpruned_conf_matrix_cv, False, is_clean, True, folder)

    Model.resultsToCSV(pruned_results_cv, True, is_clean, True, folder)
    Model.confusionMatrixToCSV(pruned_conf_matrix_cv, True, is_clean, True, folder)

    ConfMatrixUnpruned_t, ConfMatrixPruned_t, unpruned_results_t, pruned_results_t = Model.test(models, test_df)

    Model.resultsToCSV(unpruned_results_t, False, is_clean, False, folder)
    Model.confusionMatrixToCSV(ConfMatrixUnpruned_t, False, is_clean, False, folder)

    Model.resultsToCSV(pruned_results_t, True, is_clean, False, folder)
    Model.confusionMatrixToCSV(ConfMatrixPruned_t, True, is_clean, False, folder)

    return models

def saveModels(models, is_clean):
    if(is_clean):
        name = "cleanModels"
    else:
        name = "noisyModels"

    modelFile = open(name, "wb")
    pickle.dump(models, modelFile)

dt = datetime.datetime.now().strftime("%H:%M_%d-%m")
folder = "results_" + dt
os.makedirs(folder)

clean_df = LoadAsDF("./Data/cleandata_students.mat",'x')
clean_labels_df = LoadAsDF("./Data/cleandata_students.mat",'y')
clean_df = clean_df.assign(label = clean_labels_df)

cleanModels = EvaluateModel(folder, clean_df, True)
saveModels(cleanModels, True)

noisy_df = LoadAsDF("./Data/noisydata_students.mat",'x')
noisy_labels_df = LoadAsDF("./Data/noisydata_students.mat",'y')
noisy_df = noisy_df.assign(label = noisy_labels_df)

noisyModels = EvaluateModel(folder, noisy_df, False)
saveModels(noisyModels, False)
