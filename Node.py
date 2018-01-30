import pandas as pd
import numpy as np

class Node:
    def __init__(self, df, binary_target):
        self.left = None
        self.right = None
        self.variable = None
        self.input_prob = None
        self.left_prob = None
        self.right_prob = None
        
#        we will define a classification [p,1-p] as the probability of being in class binary_target
#        this is based on the incoming distribution of the node in the training data.
        
        self.input_prob = probability(df,binary_target)
        
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
            
            self.left_prob = probability(left_df,binary_target)
            self.right_prob = probability(right_df,binary_target)

            del left_df[col_str]
#            print(left_df.iloc[:,10:30])
            del right_df[col_str]
#            if the filtered set has a uniform value for binary_target, do nothing
            if(uniform_df(left_df,binary_target) == False and left_df.shape[0]!=0):
                self.left = Node(left_df,binary_target)

            if(uniform_df(right_df,binary_target)== False and right_df.shape[0]!=0):
                self.right = Node(right_df,binary_target)
                
    
    def node_classify(self,test_df):
        
#        return a classification if you have hit a leaf node
        if(self.left==None and self.right == None):
            return return_class(self.input_prob)

#        otherwise, sort into left/right based on the value of the variable column
        if(test_df[self.variable]==0):
#           may have the case that the left node leads to nothing
            if(self.left==None):
                return return_class(self.left_prob)
            else:
                return self.left.node_classify(test_df)
        else: #now for the right node
#           may have the case that the right node leads to nothing
            if(self.right==None):
                return return_class(self.right_prob)
            else:
                return self.right.node_classify(test_df)
        
                
    def print_nodetree(self,level):
        print("Level ",level,", ",self.variable)
        if(self.left!=None):
            self.left.print_nodetree(level+1)
        if(self.right!=None):
            self.right.print_nodetree(level+1)


def return_class(prob_array):
    if(prob_array[0]>prob_array[1]):
        return True
    elif(prob_array[0]<prob_array[1]):
        return False
    else:
#              return a random yes/no if the probabilities are equal
        return bool(random.getrandbits(1))

#This is thedistribution of the binary_target for each Node, produced from the training set    
def probability(df,binary_target):
    dist = [0,0]   
    
    length = df.shape[0]
    
    if (length==0):  
        return dist

    yes = df.loc[df["label"] == binary_target].shape[0]
    no = length - yes
    
    return [yes/(yes+no),no/(yes+no)]
    

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


