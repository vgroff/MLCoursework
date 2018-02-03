import Tree
import pandas as pd
import numpy as np

class Model():
    def __init__(self, data):
        dataLabels = data[["label"]]
        #print(dataLabels)
        nTrees = data.loc[:, 'label'].value_counts().shape[0]
        #print(nTrees)
        self.trees = []
        #self.trees = [Tree.Tree(data, i) for i in range(1,nTrees+1)]
        for i in range(1, nTrees+1):
            self.trees.append(Tree.Tree(data, i))
            print("Tree", i, "done")

    def classify(self, data):
        classifications = [tree.classify(data) + [i+1] for i,tree in enumerate(self.trees)]
        result  = classifications[0] + [0]
        for index, classification in enumerate(classifications[1:]):
            if (result[0] == False):
                if (classification[0] == True):
                    result = classification
                elif (classification[1] <= result[1]):
                    result = classification 
            else:
                if (classification[0] == True and classification[1] > result[1]):
                    result = classification
        return result[2]
        

    

def test_sets(input_data):
    
#    we use a 10-fold cross-validation process
    num_data_points = input_data.shape[0]
    k = 10
    test_size = num_data_points // k
    test_rows = max(1,test_size)
    
    
#    now create an array of arrays which will hold the individual test rows
    test_arrays = [[]]
    array_count = 0
    row = 0
    
    while (row < num_data_points):
        if (row!=0 and row % test_rows == 0 ):
            array_count+=1
            test_arrays.append([])
        test_arrays[array_count].append(row)
        row +=1

    # clean up the final remaining points if it is less than half the standard test size
    
    if(len(test_arrays[-1]) < (0.5 * test_rows)):
        for x in range (len(test_arrays[-1])):
            test_arrays[-2].append(test_arrays[-1][x])
        del test_arrays[-1]
    
    return test_arrays
    
def confusion_matrix(predicted,actual):
    
    size = (actual.value_counts()).shape[0]
    conf_matrix = pd.DataFrame(np.zeros((size,size),int))
    
    data_num = actual.shape[0]
    #print(actual)
    for i in range(0,data_num):
        conf_matrix[actual.iloc[i]-1][predicted[i]-1] += 1
    
    return conf_matrix



def crossValidate(data):
    test_array = test_sets(data)
    nFolds = len(test_array)
    size = data.loc[:, 'label'].value_counts().shape[0]
    totalConfMatrix = pd.DataFrame(np.zeros((size, size), int))
    for i in range(nFolds):
        validationFold = data.ix[test_array[i][0]:test_array[i][-1], :]
        if (i == 0):
            trainingFold = data.ix[test_array[1][0]:test_array[-1][-1], :]
        elif (i == nFolds - 1):
            trainingFold = data.ix[test_array[0][0]:test_array[-2][-1], :]
        else:
            trainingFold1 = data.ix[test_array[0][0]:test_array[i-1][-1], :]
            trainingFold2 = data.ix[test_array[i+1][0]:test_array[-1][-1], :]
            trainingFold  = trainingFold1.append(trainingFold2)
        #with pd.option_context('display.max_rows', 3, 'display.max_columns', 3):
        #    print(trainingFold)
        model = Model(trainingFold)
        predicted = [model.classify(validationFold.iloc[i,:]) for i in range(len(validationFold))]
        confMatrix = confusion_matrix(predicted, validationFold.loc[:, "label"])
        totalConfMatrix += confMatrix
    return totalConfMatrix
 
