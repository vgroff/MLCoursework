# -*- coding: utf-8 -*-
from numpy import *
from scipy.io import *
import numpy as np
import pandas as pd
import Node

class Tree:

    def __init__(self,df,binary_target):
        self.root_node = Node.Node(df,binary_target)
        self.binary_target = binary_target
    # Classify a data point
    def classify(self,test_df):
        classification,probability = self.root_node.node_classify(test_df)
        return [classification,probability]
    # Prune the tree
    def prune_tree(self, validation_df):
        unpruned_accuracy = self.classify_block(validation_df)
        print("prunning tree")
        self.prune(self.root_node,validation_df,unpruned_accuracy)

    def prune(self, node, validation_df, best_accuracy):
        if(node.children[0].classification != None and node.children[1].classification != None):
            children = node.children
            variable = node.variable
            node.prunning_change()

            # calculate validation accuracy of pruned tree
            pruned_accuracy = self.classify_block(validation_df)

            if(pruned_accuracy < best_accuracy):
                # revert pruning change
                node.children = children
                node.variable = variable
                node.input_prob = None
                node.classification = None
                node.pruned = False
            else:
                best_accuracy = pruned_accuracy
                if(node.prune_attempted == False):
                    self.prune(node,validation_df, best_accuracy)
        else:
            if(node.children[0].classification == None):
                self.prune(node.children[0],validation_df, best_accuracy)
            if(node.children[1].classification == None):
                self.prune(node.children[1],validation_df, best_accuracy)


    def classify_block(self, df):
        tp = 0
        tn = 0
        for i in range(0,df.shape[0]):
            classication =  self.classify(df.iloc[i,:])[0]
            if(classication and self.binary_target == df.iloc[i]['label']):
                tp += 1
            if(classication == False and self.binary_target != df.iloc[i]['label']):
                tn += 1
        accuracy = float(tp+tn)/float(df.shape[0])
        return accuracy


def print_tree(this_node,indent='', direction ='level'):

    children = this_node.children
    child_count = lambda node: count_children(this_node)
    size_branch = {child: child_count(child) for child in children}
    if (this_node.variable!=None):
        name = str(this_node.variable)
    elif (this_node.classification!=None):
        name = str(this_node.classification)

    # fill in the children
    upwards,downwards = [],[]
    if(children!=[]):
        upwards = [children[0]]
        downwards = [children[1]]
        next_indent = '{}{}{}'.format(indent, ' ' if (direction== 'u' or direction=='level') else '│', " " * len(name))
        print_tree(children[0], indent=next_indent, direction='u')

    # print the lines out of the current shape
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


def write_tree_to_file(node, tree_num, folder, is_pruned):
    if(is_pruned):
        name = "/pruned_tree"
    else:
        name = "/unpruned_tree"

    treefile = open(folder + name + str(tree_num) + ".txt", "w")
    write_tree(node,treefile)
    treefile.close()

def write_tree(this_node, treefile, indent='', direction ='level'):

    children = this_node.children
    child_count = lambda node: count_children(this_node)
    size_branch = {child: child_count(child) for child in children}
    if (this_node.variable!=None):
        name = str(this_node.variable)
    elif (this_node.classification!=None):
        name = str(this_node.classification)

    # fill in the children
    upwards,downwards = [],[]
    if(children!=[]):
        upwards = [children[0]]
        downwards = [children[1]]
        next_indent = '{}{}{}'.format(indent, ' ' if (direction== 'u' or direction=='level') else '│', " " * len(name))
        write_tree(children[0], treefile, indent=next_indent, direction='u')

    # print the lines out of the current shape
    if direction == 'u': begin = '┌'
    elif direction == 'd': begin = '└'
    else: begin = ' '

    if upwards: finish = '┤'
    elif downwards: finish = '┐'
    else: finish = ''

    # print('{}{}{}{}'.format(indent, begin, name, finish))
    line = '{}{}{}{}'.format(indent, begin, name, finish)
    treefile.write(line)
    treefile.write("\n")

    if(children!=[]):
        next_indent = '{}{}{}'.format(indent, ' ' if (direction== 'd' or direction=='level') else '│', " " * len(name))
        write_tree(children[1], treefile, indent=next_indent, direction='d')


def count_children(current_node):
    child_count = 0
    if(type(current_node)==Node.Node):
        child_count += 1
        for i in current_node.children:
            child_count += count_children(i)
    return child_count
