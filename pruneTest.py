from numpy import *
from scipy.io import *
import pandas as pda
import numpy as np
import Model
import Tree
from sklearn.model_selection import train_test_split
import copy


def Load_panda(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    output = pda.DataFrame(array)
    return output

clean_df = Load_panda("./Data/cleandata_students.mat",'x')
# short_df = clean_df.iloc[:,0:2]
clean_labels_df = Load_panda("./Data/cleandata_students.mat",'y')

clean_df = clean_df.assign(label = clean_labels_df)

train_val_df, test_df = train_test_split(clean_df, test_size=0.3)
train_df, val_df = train_test_split(train_val_df, test_size=0.3)
tree = Tree.Tree(train_df,1)
tree_copy = copy.deepcopy(tree)
# Tree.print_tree(tree.root_node)
tree.prune_tree(val_df)
# Tree.print_tree(tree.root_node)
unpruned_accuracy = tree_copy.classify_block(test_df)
pruned_accuracy = tree.classify_block(test_df)
print("Unpruned Accuracy: ", unpruned_accuracy, " pruned accuracy ", pruned_accuracy)
