import pandas as pd
import numpy as np

class Node:
    def __init__(self, df, binary_target):
        self.left = None
        self.right = None
        self.variable = None
   
#        check if the target variable entropy is 0
        incoming_entropy = entropy(df,binary_target)
        if(incoming_entropy==0):
            return
 #        stop when only the class label column remains        
        if(len(df.columns) > 1):
                
            col_str,outgoing_entropy = info_gain(df,binary_target)
            self.variable = col_str
            
            if(outgoing_entropy >= incoming_entropy):
                return
            
            left_df = df.loc[df[col_str] == 0]
            right_df = df.loc[df[col_str] == 1]

            del left_df[col_str]
#            print(left_df.iloc[:,10:30])
            del right_df[col_str]
#            if the filtered set has a uniform value for binary_target, do nothing
            if(uniform_df(left_df,binary_target) == False and left_df.shape[0]!=0):
                self.left = Node(left_df,binary_target)

            if(uniform_df(right_df,binary_target)== False and right_df.shape[0]!=0):
                self.right = Node(right_df,binary_target)
                
    def print_tree(self,level):
        print("Level ",level,", ",self.variable)
        if(self.left!=None):
            self.left.print_tree(level+1)
        if(self.right!=None):
            self.right.print_tree(level+1)



def info_gain(df, binary_target):
#    print("incoming data frame: ",df)
    min_col = None
    min_entropy = None
#    print("incoming entropy: ",entropy(df,binary_target))
#    generate column list
    column_list = list(df)
#    for all columns in the dataframe except for the class label column
    for i in range(len(column_list)-1):
        
        one_df = df.loc[df[column_list[i]] == 1]
        zero_df = df.loc[df[column_list[i]] == 0]

        one_count = one_df.shape[0]
        zero_count = zero_df.shape[0]
        
#        print("info gain: column(",column_list[i],"), one_count(",one_count,"), zero_count(",zero_count,")")
        
        one_entropy = entropy(one_df,binary_target)
        zero_entropy = entropy(zero_df,binary_target)
#       ws is the remaining entropy in the system after filtering by the column
        ws = (one_count*one_entropy + zero_count*zero_entropy)/(one_count+zero_count)
#        print("ws", ws, " i",i)
        if((min_entropy == None) or (ws < min_entropy)):
            min_entropy = ws
            min_col = column_list[i]

#    print("min_col: ",min_col,"min entropy: ",min_entropy)
    return min_col,min_entropy

def entropy(df, binary_target):
    length = df.shape[0]
    
    if (length==0):  
        return 0

    yes = df.loc[df["label"] == binary_target].shape[0]
    no = length - yes
    
    if(yes!=0):
        yes_cont = -((yes/length) * np.log2(yes/length))
    else:
        yes_cont = 0
    
    if(no!=0):
        no_cont = -((no/length) * np.log2(no/length))
    else:
        no_cont = 0
    
#    if(yes_cont+ no_cont ==1):
#        print(df)
    
    return  yes_cont + no_cont


def uniform_df(df,binary_target):
    if(entropy(df,binary_target)==0):
        return True
    return False
