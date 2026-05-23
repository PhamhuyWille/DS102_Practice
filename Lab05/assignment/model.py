import numpy as np
import random

class KMeanClustering:
    def __init__(self, n_cluster=3, max_iters=100, tolerance=10e-4):
        self.n_cluster = n_cluster
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroid = None
        
    def bayesian(self, a):
        distance = []
        for i in range(self.n_cluster):
            dis = np.sqrt(np.sum((a - self.centroid[i]) ** 2))
            distance.append(dis)
            
        return np.argmin(distance)
    
    def fit(self, X):
        random.shuffle(X)
        self.centroid = X[:self.n_cluster]
        
        for i in range(self.max_iters):
            labels = [self.bayesian(x) for x in X]
            labels = np.array(labels)
            
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else self.centroid[j]
                for j in range(self.n_cluster)
            ])

            if np.all(np.linalg.norm(new_centroids - self.centroid, axis=1) < self.tolerance):
                print(f"Converged early at iteration {i} \n")
                break
                
            self.centroid = new_centroids

        return labels