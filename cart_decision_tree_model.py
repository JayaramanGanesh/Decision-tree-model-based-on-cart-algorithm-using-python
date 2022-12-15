from sklearn.datasets import load_iris
import numpy as np


#data set loading
iris = load_iris()

#writa funcation on feature selection valu based on class node
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.left = None
        self.right = None
        self.feature_index = 0
        self.threshold = 0


#cart alg loding function 
class PJ_Cart_Tree:
    def __init__(self, max_depth, acceptable_impurity):
        self.max_depth = max_depth
        self.acceptable_impurity = acceptable_impurity



    #predict the loading data feature   
    def predict(self, inputs):
        current_node = self.tree
        while current_node.left:
            if inputs[current_node.feature_index] < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.predicted_class



    #fit the value in classifications  
    def fit(self, x, y):
        self.classifications = len(set(y))
        self.features = x.shape[1]
        self.tree = self.create_tree(x, y)
        

    #data impurity in gini
    def gini_impurity(y):
        instances = np.bincount(y)
        total = np.sum(instances)
        p = instances/total
        return 1.0 - np.sum(np.power(p,2)) 
    
    #spliting the chart
    def cart_split(self, x, y):
        m = y.size
        if m <= 1:
            return None, None
        best_index = None
        best_threshold = None
        parent = [np.sum(y == c) for c in range(self.classifications)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in parent)

        if best_gini >= self.acceptable_impurity:
            for index in range(self.features):
                thresholds, classes = zip(*sorted(zip(x[:, index], y)))
                num_left = [0] * self.classifications
                num_right = parent.copy()
                for i in range(1, m):
                    c = classes[i - 1]
                    num_left[c] += 1
                    num_right[c] -= 1
                    gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.classifications))
                    gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.classifications))
                    gini = (i * gini_left + (m - i) * gini_right) / m
                    if thresholds[i] == thresholds[i - 1]:
                        continue
                    if gini < best_gini:
                        best_gini = gini
                        best_index = index
                        best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return best_index, best_threshold
    
    #build on cart decision tree
    def create_tree(self, x, y, depth=0):
        samples_class = [np.sum(y == i) for i in range(self.classifications)]
        predicted_class = np.argmax(samples_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            index, thr = self.cart_split(x, y)
            if index is not None:
                indices_left = x[:, index] < thr
                x_left = x[indices_left]
                y_left = y[indices_left]
                x_right = x[~indices_left]
                y_right = y[~indices_left]
                node.feature_index = index
                node.threshold = thr
                node.left = self.create_tree(x_left, y_left, depth + 1)
                node.right = self.create_tree(x_right, y_right, depth + 1)
        return node



#load the data from model
tree = PJ_Cart_Tree(max_depth=4, acceptable_impurity=0.2)
tree.fit(iris.data, iris.target)
print(iris.data[4])

#predit the value
cl = tree.predict(iris.data[4])
print('Classified as {}'.format(iris.target_names[cl]))