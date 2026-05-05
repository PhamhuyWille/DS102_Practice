import numpy as np

class Node:
    def __init__ (self, gini, feature, threshold, left, right, value):
        self.gini = gini
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreescratch:
    def __init__(self, depth: int = 10, min_sample_split: int=2):
        self.depth = depth
        self.min_sample_split = min_sample_split
        self.root = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = len(np.unique(y))
        self.root = self.build_tree(X, y, 0)

    def predict_one(self, x, node: Node):
        if node.value != None:
            return node.value
        
        feature = node.feature
        if x[feature] <=node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return [self.predict_one(x, self.root) for x in X]
    
    def major_vote(self, y):
        labels, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        value = labels[index]
        return value

    def build_tree(self, X, y, depth):
        n, features = X.shape
        num_class = len(np.unique(y))

        if (
            (depth >= self.depth) or 
            (num_class == 1) or 
            (n < self.min_sample_split)
        ):
            return Node(
                gini=None,
                feature=None,
                threshold=None,
                left=None,
                right=None,
                value=self.major_vote(y)
            )
        
        gini, feature, threshold = self.best_split(X, y)
        if feature is None:
            return Node(
                gini=None,
                feature=None,
                threshold=None,
                left=None,
                right=None,
                value=self.major_vote(y)
            )
        left_mask  = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left, X_right = X[left_mask], X[right_mask]
        y_left, y_right = y[left_mask], y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return Node(
                gini=None,
                feature=None,
                threshold=None,
                left=None,
                right=None,
                value=self.major_vote(y)
            )
        
        left_child  = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        return Node(
            gini = gini,
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
            value=None
        )

    def gini(self, X, y, threshold):
        left_mask = X <= threshold
        right_mask = X > threshold
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        num_left = len(y_left)
        num_right = len(y_right)
        
        _, freq_left = np.unique(y_left, return_counts=True)
        _, freq_right = np.unique(y_right, return_counts=True)
        
        if num_left == 0 or num_right == 0:
            return 1

        freq_left = freq_left / num_left
        freq_right = freq_right / num_right
        
        gini_left = 1 - np.sum(freq_left * freq_left)
        gini_right = 1 - np.sum(freq_right * freq_right)

        gini_i = (num_left / (num_left + num_right)) * gini_left + (num_right / (num_left + num_right)) * gini_right
        return gini_i

    def best_split(self, X, y):
        n, features = X.shape
        if n < 2:
            return None, None, None
        
        perfect_gini = np.inf
        perfect_threshold = None
        perfect_feature = None

        for feature in range(features):
            
            sort_index = np.argsort(X[:, feature])
            X_sorted = X[sort_index, feature]
            y_sorted = y[sort_index]
            
            for i in range(n - 1):
                if X_sorted[i] == X_sorted[i + 1]:
                    continue

                threshold = (X_sorted[i] + X_sorted[i + 1]) / 2
                g = self.gini(X_sorted, y_sorted, threshold)
                
                if g < perfect_gini:
                    perfect_gini = g
                    perfect_feature = feature
                    perfect_threshold = threshold

        return perfect_gini, perfect_feature, perfect_threshold