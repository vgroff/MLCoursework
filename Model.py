import Tree
import pandas as pd
import numpy as np
import copy

class Model():
    def __init__(self, data):
        # Count the number of different values, representing each tree
        #print(data.loc[:, 'label'].value_counts())
        nTrees = 1#data.loc[:, 'label'].value_counts().shape[0]
        # Build list nTrees long
        self.trees = []
        self.pruned_trees = []
        for i in range(1, nTrees+1):
            self.trees.append(Tree.Tree(data, i))
            print("Tree", i, "done")

    def classify(self, data, is_pruned):
        # Classify each of the data points with each of the trees, giving [T/F,Probability,Emotion]
        if(is_pruned):
            classifications = [tree.classify(data) + [i+1] for i,tree in enumerate(self.pruned_trees)]
        else:
            classifications = [tree.classify(data) + [i+1] for i,tree in enumerate(self.trees)]
        # Starting with the first classification, try to find the optimal one
        result = classifications[0]
        for index, classification in enumerate(classifications[1:]):
            if (result[0] == False):
                # If the current best result is False
                if (classification[0] == True):
                    # If this classification is True, it is the best result so far
                    result = classification
                elif (classification[1] <= result[1]):
                    # If this classication is False but with a lower probability, it is better
                    result = classification
            elif (classification[0] == True and classification[1] > result[1]):
                # If this classification is True with a higher probability, it is better
                result = classification
        return result[2]

    def rawClassify(self, data):
        return [tree.classify(data) + [i+1] for i,tree in enumerate(self.trees)]

    def prune(self, prune_df):
        for i in range(0, len(self.trees)):
            tree_copy = copy.deepcopy(self.trees[i])
            tree_copy.prune_tree(prune_df)
            self.pruned_trees.append(tree_copy)

    def print_to_file(self, folder, is_pruned, is_clean):
        for i in range(0, len(self.trees)):
            # Tree.print_tree(self.trees[i].root_node)
            if(is_pruned):
                Tree.write_tree_to_file(self.pruned_trees[i].root_node, i+1, folder, is_pruned, is_clean)
            else:
                Tree.write_tree_to_file(self.trees[i].root_node, i+1, folder, is_pruned, is_clean)

    def test_model(self, test_df):
        prediction = []
        for i in range(0,test_df.shape[0]):
            prediction.append(self.classify(test_df.iloc[i,:], False))
        return prediction

def test_sets(input_data):

    # we use a 10-fold cross-validation process
    num_data_points = input_data.shape[0]
    k = 10
    test_size = num_data_points // k
    test_rows = max(1,test_size)


    # now create an array of arrays which will hold the individual test rows
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

def get_test_dfs(input_data, k):
    # we use a 10-fold cross-validation process
    test_dfs = []
    for i in range(k):
        dfs = split(1/(k-i), input_data)
        test_dfs.append(dfs[0])
        input_data = dfs[1]
    return test_dfs

def confusion_matrix(predicted,actual):

    size = (actual.value_counts()).shape[0]
    conf_matrix = pd.DataFrame(np.zeros((size,size),int))

    data_num = actual.shape[0]
    for i in range(0,data_num):
        conf_matrix[actual.iloc[i]-1][predicted[i]-1] += 1

    return conf_matrix

def crossValidate(data, k, folder, is_clean):
    # Get the folds
    data = data.sample(frac=1).reset_index(drop=True)
    test_array = get_test_dfs(data, k)
    nFolds = len(test_array)
    models = []

    # Make an array the size of the nunber of labels for the confusion matrix
    size = data.loc[:, "label"].value_counts().shape[0]
    totalConfMatrixUnpruned = pd.DataFrame(np.zeros((size, size), int))
    totalConfMatrixPruned = pd.DataFrame(np.zeros((size, size), int))
    for i in range(nFolds-1):
        # Validation fold is the ith fold
        validationFold = test_array[i]
        pruneFold = test_array[i+1]
        # Put together the rest as the training fold
        trainingFolds = list(test_array)
        del trainingFolds[i]
        del trainingFolds[i]
        trainingFold = trainingFolds[0]

        for i in range(1, k-2):
            trainingFold = trainingFold.append(trainingFolds[i])
        # Build model with training fold and get the predictions on the validation fold,
        # then build the confusion matrix
        model = Model(trainingFold)
        models.append(model)

        model.prune(pruneFold)

        model.print_to_file(folder, True, is_clean)
        model.print_to_file(folder, False, is_clean)

        unpruned_predicted = [model.classify(validationFold.iloc[i,:], False) for i in range(len(validationFold))]
        pruned_predicted = [model.classify(validationFold.iloc[i,:], True) for i in range(len(validationFold))]

        confMatrixUnpruned = confusion_matrix(unpruned_predicted, validationFold.loc[:, "label"])
        totalConfMatrixUnpruned += confMatrixUnpruned

        confMatrixPruned = confusion_matrix(pruned_predicted, validationFold.loc[:, "label"])
        totalConfMatrixPruned += confMatrixPruned


    return models, totalConfMatrixUnpruned, totalConfMatrixPruned

def split(proportion, data):
    num_data_points = data.shape[0]
    dataPoints1 = int(num_data_points*proportion)
    data1 = data.iloc[0:dataPoints1, :]
    data2 = data.iloc[dataPoints1:num_data_points, :]
    return [data1, data2]

def performanceMetrics(confMatrix):
    # Here is where the classification measures are calculated
    # Number one: total classification rate.
    total_predictions = confMatrix.values.sum()
    total_sum = 0
    for row in range(0,6):
        total_sum = total_sum + confMatrix.iloc[row].loc[row]

    accuracy = (total_sum/total_predictions)*100
    print("Total Predictions:",total_predictions)
    print("Total Correct Predictions:",total_sum)
    print("Classification Rate / Accuracy:", "{0:.0f}%".format(accuracy,"\n"))

    # Number two: class specific classification measures
    unweighted_average_recall = 0
    precisions = []
    recalls = []
    F1s = []
    for class_number in range(0,6):
        print("Classification measures for class",class_number,":")
        number_correct = confMatrix.iloc[class_number].loc[class_number]
        total_number_of_class = confMatrix.iloc[class_number].sum()
        total_number_labelled = confMatrix[class_number].sum()

        precision = number_correct / total_number_labelled
        precisions.append(precision)
        recall = number_correct / total_number_of_class
        recalls.append(recall)
        F1 = 2*((precision*recall)/(precision+recall))
        F1s.append(F1)
        unweighted_average_recall = unweighted_average_recall + recall

        print("Precision:","{0:.0f}%".format(precision*100))
        print("Recall:","{0:.0f}%".format(recall*100))
        print("F1:","{0:.0f}%".format(F1*100),"\n")

    unweighted_average_recall = unweighted_average_recall / 6
    print("Unweighted Average Recall:","{0:.0f}%".format(unweighted_average_recall*100),"\n")
    results = {"accuracy":accuracy, "precision":precisions, "recall":recalls,
     "F1":F1s, "uneweighted_average_recall":unweighted_average_recall}
    return results
    #return [accuracy, precision, recall, F1, unweighted_average_recall]

def performanceMetricsDF(confMatrix):
    # Here is where the classification measures are calculated
    # Number one: total classification rate.
    total_predictions = confMatrix.values.sum()
    total_sum = 0
    for row in range(0,6):
        total_sum = total_sum + confMatrix.iloc[row].loc[row]

    accuracy = (total_sum/total_predictions)*100
    print("Total Predictions:",total_predictions)
    print("Total Correct Predictions:",total_sum)
    print("Classification Rate / Accuracy:", "{0:.0f}%".format(accuracy,"\n"))

    # Number two: class specific classification measures
    unweighted_average_recall = 0
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for class_number in range(0,6):
        print("Classification measures for class",class_number,":")
        number_correct = confMatrix.iloc[class_number].loc[class_number]
        total_number_of_class = confMatrix.iloc[class_number].sum()
        total_number_labelled = confMatrix[class_number].sum()

        # accuracies = accuracy
        precision = number_correct / total_number_labelled
        precisions.append(precision)
        recall = number_correct / total_number_of_class
        recalls.append(recall)
        F1 = 2*((precision*recall)/(precision+recall))
        F1s.append(F1)
        unweighted_average_recall = unweighted_average_recall + recall

        print("Precision:","{0:.0f}%".format(precision*100))
        print("Recall:","{0:.0f}%".format(recall*100))
        print("F1:","{0:.0f}%".format(F1*100),"\n")

    unweighted_average_recall = unweighted_average_recall / 6
    print("Unweighted Average Recall:","{0:.0f}%".format(unweighted_average_recall*100),"\n")
    results = {"Accuracy":accuracy, "Unweighted Avg. Recall":unweighted_average_recall, "Precision":precisions, "Recall":recalls, "F1":F1s}

    # resultsToCSV(results, is_pruned, is_clean, True, folder)
    # confusionMatrixToCSV(confMatrix, is_pruned, is_clean, True, folder)

    return results

def resultsToCSV(results, is_pruned, is_clean, is_cross_val, folder):
    results_df = pd.DataFrame(results, index=['1', '2', '3', '4', '5', '6'])
    results_df = results_df.round(2)

    filename = "/"
    if(is_pruned):
        filename = filename + "pruned_"
    else:
        filename = filename + "unpruned_"

    if(is_clean):
        filename = filename + "clean_"
    else:
        filename = filename + "noisy_"

    if(is_cross_val):
        filename = filename + "cross_val_"
    else:
        filename = filename + "test_"

    filename = filename + "table"

    results_df.to_csv(folder + filename + ".csv", sep='\t')

    print(results_df)

def confusionMatrixToCSV(confMatrix, is_pruned, is_clean, is_cross_val, folder):
    filename = "/"
    if(is_pruned):
        filename = filename + "pruned_"
    else:
        filename = filename + "unpruned_"

    if(is_clean):
        filename = filename + "clean_"
    else:
        filename = filename + "noisy_"

    if(is_cross_val):
        filename = filename + "cross_val_"
    else:
        filename = filename + "test_"

    filename = filename + "matrix"

    confMatrix.to_csv(folder + filename + ".csv", sep='\t')

def test(models, test_df):

    size = test_df.loc[:, "label"].value_counts().shape[0]
    totalConfMatrixUnpruned = pd.DataFrame(np.zeros((size, size), int))
    totalConfMatrixPruned = pd.DataFrame(np.zeros((size, size), int))

    for model in models:
        unpruned_predicted = [model.classify(test_df.iloc[i,:], False) for i in range(len(test_df))]
        pruned_predicted = [model.classify(test_df.iloc[i,:], True) for i in range(len(test_df))]

        confMatrixUnpruned = confusion_matrix(unpruned_predicted, test_df.loc[:, "label"])
        totalConfMatrixUnpruned += confMatrixUnpruned

        confMatrixPruned = confusion_matrix(pruned_predicted, test_df.loc[:, "label"])
        totalConfMatrixPruned += confMatrixPruned

    unpruned_results  = performanceMetricsDF(confMatrixUnpruned)
    pruned_results = performanceMetricsDF(confMatrixPruned)

    return totalConfMatrixUnpruned, totalConfMatrixPruned, unpruned_results, pruned_results
