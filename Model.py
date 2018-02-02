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
        result = []
        for tree in self.trees:
            result.append(tree.classify(data))
        return result
        
