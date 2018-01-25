import pandas as pd
import numpy as np

class Node:
    def __init__(self, df, binary_target):
        self.left = None
        self.right = None

        if(len(df.columns) != 1):
            col_str = info_gain(df,binary_target)
            left_df = df.loc[df['col_str'] == 0]
            right_df = df.loc[df['col_str'] == 1]

            left_df.drop(columns=[col_str])
            right_df.drop(columns=[col_str])

            if(uniform_df(left_df) == False):
                self.left = Node(left_df,binary_target)

            if(uniform_df(right_df)== False):
                self.right = Node(right_df,binary_target)



def info_gain(df, binary_target):
    # yes = [0 for i in df.columns]
    # no = [0 for i in df.columns]
    min_col = None
    min_entropy = None
    for i in df.columns:
        one_df = df.loc[df[i] == 1]
        zero_df = df.loc[df[i] == 0]

        one_count = one_df.shape[0]
        zero_count = zero_df.shape[0]

        one_entropy = entropy(one_df,binary_target)
        zero_entropy = entropy(zero_df,binary_target)

        ws = (one_count*one_entropy + zero_count*zero_entropy)/(one_count+zero_count)
        print("ws", ws)
        if(min_entropy == None or ws <min_entropy):
            min_entropy = ws
            min_col = i

    print("min_col",min_col)
    return min_col

def entropy(df, binary_target):
    length = df.shape[0]
    yes = df.loc[df["label"] == binary_target].shape[0]
    no = length - yes
    return  -(yes/length) * np.log2(yes/length) - (no/length) * np.log2(no/length)

def uniform_df(df):
    if(df.loc[:,"label"].var() == 0):
        return True
    return False
