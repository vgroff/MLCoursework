from numpy import *
from scipy.io import *
import pandas as pda
import numpy as np
import Model
import Tree
from sklearn.model_selection import train_test_split

def Load_panda(Filename, Variable):
    mat = loadmat(Filename)
    array = mat[Variable]
    output = pda.DataFrame(array)
    return output

clean_df = Load_panda("./Data/cleandata_students.mat",'x')
clean_labels_df = Load_panda("./Data/cleandata_students.mat",'y')
clean_df = clean_df.assign(label = clean_labels_df)

train_df, test_df = train_test_split(clean_df, test_size=0.2)
model = Model.Model(train_df)
# for i in range(0,test_df.shape[1]):
#     classication,prob = tree.classify(test_df.iloc[4,:])
#     print(classication, prob, test_df.iloc[i]['label'])
result = model.classify(test_df.iloc[1,:])
print(result, test_df.iloc[1]['label'])
