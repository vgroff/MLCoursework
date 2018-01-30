from numpy import *
from scipy.io import *
import numpy as np
import pandas as pda
import Node


class Tree:
    def __init__(self,df):
        
        self.root_nodes = []
        for i in range(0,6):
            self.root_nodes.append(Node.Node(df,i+1))
    
#    def print_tree(self):
#        self.root_node.print_nodetree(0)
        
#    function to recurse down an entire tree and remove and delete any nodes that are empty.
    
    def classify(self,test_df):
        classification = [0 for i in range(6)]
        
        for j in range(0,6):
            classification[j] = self.root_nodes[j].node_classify(test_df)
        return classification
        

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
    
    if(len(test_arrays[len(test_arrays)-1]) < (0.5 * test_rows)):
        for x in range (len(test_arrays[len(test_arrays)-1])):
            test_arrays[len(test_arrays)-2].append(test_arrays[len(test_arrays)-1][x])
        del test_arrays[len(test_arrays)-1]
    
    return test_arrays
    
def confusion_matrix(predicted,actual):
    
    size = (actual.value_counts()).shape[0]
    array = np.zeros((size,size),int)
    conf_matrix = pda.DataFrame(array)
    
    data_num = actual.shape[0]
    
    for i in range(0,data_num):
        conf_matrix[actual[i]-1][predicted[i]-1] += 1
    
    return conf_matrix
