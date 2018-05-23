from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt 

def find_clusters(X, n_clusters, rseed=2):
    ## Choose random clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    ## This will give you n_clusters number of centers
    ## which will have the dimensions of your dataset
    centers = X[i]

    while True:
        ## Assign labels based on closest center 
        labels = pairwise_distances_argmin(X, centers)

        ## Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
    
        ## Check for convergence 
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels 

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

centers, labels = find_clusters(X,4)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis')

x = np.random.binomial(19,0.33, size=(200,3))

centers, labels = find_clusters(X,4)
plt.scatter(x[:, 0], x[:, 1], c=labels,
            s=50, cmap='viridis')