import numpy as np
from numpy.linalg import norm
from sys import float_info


def loss(A, B, P, C=0.0, D=0.0, lam=0.0):
    """The subgraph matching loss for weighted undirected (and
    unlabeled) graphs.

    A and B are the adjacency matrices, P is a partial permutation
    matrix, C and D are k-dimensional node labels, and lam(bda) is the
    parameter to balance graph weights and node labels.

    """
    return 0.5 * norm(A - P.dot(B.dot(P.T))) + \
        lam * norm(C - P.dot(D))


def fastPFP(A, B, C=0.0, D=0.0, lam=0.0, alpha=0.5, threshold1=1.0e-6,
            threshold2=1.0e-6, X=None, Y=None, verbose=True, max_iter1=100,
            max_iter2=100):
    """The fastPFP algorithm for the subgraph matching problem, as
    proposed in the paper 'A Fast Projected Fixed-Point Algorithm for
    Large Graph Matching' by Yao Lu, Kaizhu Huang, Cheng-Lin Liu.

    See: http://arxiv.org/abs/1207.1114

    Note: in the paper A, B, C and D are called A, A', B and B'.
    """
    size1 = A.shape[0]
    size2 = B.shape[0]
    one1 = np.ones((size1, 1))
    one2 = np.ones((size2, 1))
    if X is None:
        X = one1.dot(one2.T) / (size1 * size2)

    if Y is None:
        Y = np.zeros((size1, size1))

    K = np.atleast_2d(C).dot(np.atleast_2d(D).T)

    float_max = float_info.max
    epsilon1 = epsilon2 = float_max
    iter1 = 0
    while epsilon1 > threshold1 and iter1 < max_iter1:
        Y[:size1, :size2] = A.dot(X.dot(B)) + lam * K
        epsilon2 = float_max
        iter2 = 0
        while epsilon2 > threshold2 and iter2 < max_iter2:
            tmp = np.eye(size1, size1) / size1
            tmp += (one1.T.dot(Y.dot(one1)) / (size1 * size1)) \
                   * (np.eye(size1, size1))
            tmp -= Y / size1
            tmp = tmp.dot(one1.dot(one1.T))
            Y_new = Y + tmp - one1.dot(one1.T.dot(Y)) / size1
            Y_new = (Y_new + np.abs(Y_new)) / 2.0
            epsilon2 = np.abs(Y_new - Y).max()
            Y = Y_new
            iter2 += 1

        if verbose:
            print("epsilon2 = %s" % epsilon2)

        X_new = (1.0 - alpha) * X + alpha * Y[:size1, :size2]
        X_new = X_new / X_new.max()
        epsilon1 = np.abs(X_new - X).max()
        X = X_new
        if verbose:
            print("epsilon1 = %s" % epsilon1)
            loss_X = loss(A, B, X, C, D)
            print("Loss(X) = %s" % loss_X)

        iter1 += 1

    return X


def fastPFP_faster(A, B, C=0.0, D=0.0, lam=0.0, alpha=0.5, threshold1=1.0e-6,
                   threshold2=1.0e-6, X=None, Y=None, verbose=True,
                   max_iter1=100, max_iter2=100):
    """A faster and more efficient implementation of fastPFP().
    """
    size1 = A.shape[0]
    size2 = B.shape[0]
    if X is None:
        X = np.ones((size1, size2)) / (size1 * size2)

    if Y is None:
        Y = np.zeros((size1, size1))

    K = np.atleast_2d(C).dot(np.atleast_2d(D).T)

    float_max = float_info.max
    epsilon1 = epsilon2 = float_max
    iter1 = 0
    while epsilon1 > threshold1 and iter1 < max_iter1:
        Y[:size1, :size2] = A.dot(X.dot(B)) + lam * K
        epsilon2 = float_max
        iter2 = 0
        while epsilon2 > threshold2 and iter2 < max_iter2:
            tmp = (1.0 + Y.sum() / size1 - Y.sum(1)) / size1
            Y_new = Y + tmp[:, None] - Y.sum(0) / size1
            Y_new = np.clip(Y_new, 0.0, float_max)
            epsilon2 = np.abs(Y_new - Y).max()
            Y = Y_new
            iter2 += 1

        if verbose:
            print("epsilon2 = %s" % epsilon2)

        X_new = (1.0 - alpha) * X + alpha * Y[:size1, :size2]
        X_new = X_new / X_new.max()
        epsilon1 = np.abs(X_new - X).max()
        X = X_new
        if verbose:
            print("epsilon1 = %s" % epsilon1)

        iter1 += 1

    return X


def greedy_assignment(X):
    """A simple greedy algorithm for the assignment problem as
    proposed in the paper of fastPFP. It creates a proper partial
    permutation matrix (P) from the result (X) of the optimization
    algorithm fastPFP.
    """
    XX = X.copy()
    min = XX.min() - 1.0
    P = np.zeros(X.shape)
    while (XX > min).any():
        row, col = np.unravel_index(XX.argmax(), XX.shape)
        P[row, col] = 1.0
        XX[row, :] = min
        XX[:, col] = min

    return P
