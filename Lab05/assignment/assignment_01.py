import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    print('=' * 80)
    print('Run assignment 01: Kịch bản Lý tưởng')
    print('=' * 80)
    data = create_data()
    model = KMeanClustering(max_iters=800)
    labels = model.fit(data)
    print('The centroids by K-means\n')
    print(model.centroid)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=data[:, 0], 
        y=data[:, 1], 
        hue=labels, 
        palette="viridis",   
        alpha=0.75, 
        s=60,               
        edgecolor="none",  
        legend="full"
    )

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis', alpha=0.7)
    plt.scatter(model.centroid[:, 0], model.centroid[:, 1], c='red', s=100, label='Centroids')
    plt.title("K-Means From Scratch Result for Assignment 1", fontsize=14, fontweight='bold')
    plt.grid('--')
    plt.legend(title="Class / Elements", loc="upper right", frameon=True)
    plt.savefig('results/Result_assignment_01.png')

if __name__ == '__main__':
    run_assignment01()