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
        self.binary_target = binary_target
    
#    def print_tree(self):
#        self.root_node.print_nodetree(0)
    
    def classify(self,test_df):
        classification,probability = self.root_node.node_classify(test_df)
        return [classification,probability]
    
    
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



