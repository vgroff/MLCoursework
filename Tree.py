from numpy import *
from scipy.io import *
import pandas as pda
import numpy as np
import Node


class Tree:
    def __init__(self,df,binary_target):
        self.root_node = None
        
        
        self.root_node.Node(df,binary_target)