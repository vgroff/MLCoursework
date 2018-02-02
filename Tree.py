# -*- coding: utf-8 -*-
from numpy import *
from scipy.io import *
from graphviz import *
import numpy as np
import pandas as pda
import Node



class Tree:
    
    def __init__(self,df,binary_target):      
        self.root_node = Node.Node(df,binary_target)
    
#    def print_tree(self):
#        self.root_node.print_nodetree(0)
    
    def classify(self,test_df):
        classification,probability = self.root_node.node_classify(test_df)
        return classification,probability
    
    
def print_tree(this_node,indent='', direction ='level'):
 
    children = this_node.children
    child_count = lambda node: count_children(this_node)
    size_branch = {child: child_count(child) for child in children}
    if (this_node.variable!=None):
        name = str(this_node.variable)
    elif (this_node.classification!=None):
        name = str(this_node.classification)

#    fill in the children
    upwards,downwards = [],[]
    if(children!=[]):
        upwards = [children[0]] 
        downwards = [children[1]]
        next_indent = '{}{}{}'.format(indent, ' ' if (direction== 'u' or direction=='level') else '│', " " * len(name))
        print_tree(children[0], indent=next_indent, direction='u')

#   print the lines out of the current shape
    if direction == 'u': begin = '┌'
    elif direction == 'd': begin = '└'
    else: begin = ' '

    if upwards: finish = '┤'
    elif downwards: finish = '┐'
    else: finish = ''

    print('{}{}{}{}'.format(indent, begin, name, finish))

    if(children!=[]):
        next_indent = '{}{}{}'.format(indent, ' ' if (direction== 'd' or direction=='level') else '│', " " * len(name))
        print_tree(children[1], indent=next_indent, direction='d')


def count_children(current_node):
    child_count = 0
    if(type(current_node)==Node.Node):
        child_count += 1
        for i in current_node.children:
            child_count += count_children(i)
    return child_count



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
