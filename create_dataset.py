import numpy as np
from scipy.spatial import distance_matrix


def create_dataset_artificial(size1, size2, same=True, sigma1=None, sigma2=None, verbose=False):
    """This function creates two adjacency matrices graphs whose
    respective number of nodes is size1 and size2, respectively.
    
    The graphs refer to 2D clouds of point where the edges, i.e. the
    values of the adjacency matrices, are similarities between points
    defined as s(x1, x2) = exp(-d(x1,x2)**2 / sigma**2) where d() is
    the Euclidean distance and sigma is either provided by the user or
    defined as the median distance between the points.

    If 'same' is True, then the smaller cloud of points is a subset of
    the larger cloud, i.e. the corresponding graphs have a perfect
    subgraph match.
    """
    print("Dateset creation.")
    if same:
        X = np.random.rand(max([size1, size2]), 2)
        X1 = X[:size1]
        X2 = X[:size2]
        dm = distance_matrix(X, X)
        dm1 = dm[:size1, :size1]
        dm2 = dm[:size2, :size2]
        sigma = np.median(dm[np.triu_indices(dm.shape[0], 1)])
        if sigma1 is None:
            sigma1 = sigma

        if sigma2 is None:
            sigma2 = sigma
            
    else:
        X1 = np.random.rand(size1, 2)
        X2 = np.random.rand(size2, 2)
        dm1 = distance_matrix(X1, X1)
        dm2 = distance_matrix(X2, X2)
        if sigma1 is None:
            sigma1 = np.median(dm1[np.triu_indices(size1, 1)])

        if sigma2 is None:
            sigma2 = np.median(dm2[np.triu_indices(size2, 1)])


    if verbose: print("create_dataset_artificial: sigma1=%s , sigma2=%s" % (sigma1, sigma2))
    A = np.exp(- dm1 * dm1 / (sigma1 ** 2))
    B = np.exp(- dm2 * dm2 / (sigma2 ** 2))

    return A, B, X1, X2
