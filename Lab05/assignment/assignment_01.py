import numpy as np
import matplotlib.pyplot as plt
from assignment.model import KMeanClustering

def create_data():
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0],
            [0, 1]]

    data = []
    for mean in means:
        cluster = np.random.multivariate_normal(mean, cov, 200)
        data.append(cluster)

    data = np.vstack(data)
    return data

def run_assignment01():
    data = create_data()
    model = KMeanClustering(max_iters=800)
    labels = model.fit(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis', alpha=0.7)
    plt.scatter(model.centroid[:, 0], model.centroid[:, 1], c='red', s=100, label='Centroids')
    plt.title("K-Means From Scratch Result")
    plt.legend()
    plt.savefig('results/Result_assignment_01.png')
    plt.show()

if __name__ == '__main__':
    run_assignment01()