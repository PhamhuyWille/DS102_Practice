import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from assignment.model import KMeanClustering

def create_data():
    cov = [[1, 0],
           [0, 1]]
    cluster_01 = np.random.multivariate_normal([2, 2], cov, 1200)
    cluster_02 = np.random.multivariate_normal([8, 3], cov, 200)
    cluster_03 = np.random.multivariate_normal([3, 6], cov, 1000)
    data = np.vstack([cluster_01, cluster_02, cluster_03])
    return data

def run_assignment02():
    print('=' * 80)
    print('Run assignment 02: Thách thức về độ lệch kích thước cụm')
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
    plt.savefig('results/Result_assignment_02.png')


if __name__ == '__main__':
    run_assignment02()