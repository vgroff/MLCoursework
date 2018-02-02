import Tree

class Model():
    def __init__(self, data):
        dataLabels = data[["label"]]
        #print(dataLabels)
        nTrees = dataLabels.max().values[0]
        #print(nTrees)
        self.trees = []
        #self.trees = [Tree.Tree(data, i) for i in range(1,nTrees+1)]
        for i in range(1, nTrees+1):
            self.trees.append(Tree.Tree(data, i))
            print("Tree", i, "done")

    def classify(self, data):
        classifications = [tree.classify(data) + [i+1] for i,tree in enumerate(self.trees)]
        result  = classifications[1] + [1]
        for index, classification in enumerate(classifications[1:]):
            if (result[0] == False):
                if (classification[0] == True):
                    result = classification
                elif (classification[1] <= result[1]):
                    result = classification 
            else:
                if (classification[0] == True and classification[1] > result[1]):
                    result = classification
        return result[2]
        
