import numpy as np
import matplotlib.pyplot as plt
from assignment.model import KMeanClustering

def create_data():
    cov_01 = [[1, 0],
              [0, 1]]
    cov_02 = [[10, 0],
              [0, 1]]

    cluster_01 = np.random.multivariate_normal([2, 2], cov_01, 200)
    cluster_02 = np.random.multivariate_normal([8, 3], cov_01, 200)
    cluster_03 = np.random.multivariate_normal([3, 6], cov_02, 200)
    data = np.vstack([cluster_01, cluster_02, cluster_03])
    return data

def run_assignment03():
    data = create_data()
    model = KMeanClustering(max_iters=800)
    labels = model.fit(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis', alpha=0.7)
    plt.scatter(model.centroid[:, 0], model.centroid[:, 1], c='red', s=100, label='Centroids')
    plt.title("K-Means From Scratch Result")
    plt.legend()
    plt.savefig('results/Result_assignment_03.png')
    plt.show()

if __name__ == '__main__':
    run_assignment03()