from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d

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

## This works on 3-D as well - and N-D data - 
## can be visualised as below 

x = np.random.normal(2, 0.03, size=(200,3))
x += np.random.normal(5, 0.02, size=(200,3))
x += np.random.normal(-4, 0.01, size=(200,3))
centers, labels = find_clusters(x,3)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels,
            s=50, cmap='viridis')