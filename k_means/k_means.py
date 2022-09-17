import numpy as np 
import pandas as pd 
import time
import random as randy
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k=2, scale = np.array([1, 1])):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.k = k
        self.scale = scale
        self.center = np.zeros(((self.k),2))
        # print(center)
        # for i in range(k):
        #     j = randy.randint(0, length())
        #     center[k] = 
        pass
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        data = X.to_numpy()
        data[:,0] *= self.scale[0]
        data[:,1] *= self.scale[1]
        addon = np.zeros((len(X['x1']),2))
        data = np.hstack((data, addon))
        for i in range(self.k):
            j = randy.randint(0, len(X['x1'])-1)
            self.center[i] = data[j][0:2]
        data[:,3] += 1000

        for i in range(len(X['x1'])):
            for j in range(self.k):
                dist = euclidean_distance(data[i,0:2], self.center[j])
                if dist < data[i,3]:
                    data[i,3] = dist
                    data[i,2] = j

        lastFC = np.zeros(((self.k),3))
        run = True
        times = 0
        while run:
            findCenter = np.zeros(((self.k),3))
            for i in range(len(data[:])):
                point = int(round(data[i,2]))
                findCenter[point,0:2] += data[i,0:2]
                findCenter[point,2] += 1
            stop = True
            for i in range(self.k):
                if abs(findCenter[i,2] - lastFC[i,2]>10):
                    stop = False
            for i in range(self.k):
                self.center[i] = findCenter[i,0:2]*(1/findCenter[i,2])
            data[:,3] += 1000
            for i in range(len(X['x1'])):
                for j in range(self.k):
                    dist = euclidean_distance(data[i,0:2], self.center[j])
                    if dist < data[i,3]:
                        data[i,3] = dist
                        data[i,2] = j
            times += 1
            lastFC = findCenter
            if times > 50:
                run = False
            if stop:
                run = False

        self.data = data
        # raise NotImplementedError()
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement 
        ret = np.zeros(len(X['x1']), int)
        for i in range(len(ret)):
            ret[i] = int(round(self.data[i,2]))
        return ret
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        # TODO: Implement 
        return self.center
    
    
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
