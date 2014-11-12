import numpy as np

def fastPFP(A, B, alpha=0.5, threshold1=1.0e-6, threshold2=1.0e-6, X=None, Y=None):
    """The fastPFP algorithm for the subgraph matching problem, as
    proposed in the paper 'A Fast Projected Fixed-Point Algorithm for
    Large Graph Matching' by Yao Lu, Kaizhu Huang, Cheng-Lin Liu.

    See: http://arxiv.org/abs/1207.1114
    """
    size1 = A.shape[0]
    size2 = B.shape[0]
    one1 = np.ones((size1, 1))
    one2 = np.ones((size2, 1))
    if X is None: X = one1.dot(one2.T) / (size1 * size2)
    if Y is None: Y = np.zeros((size1, size1))
    
    loss_best = epsilon1 = epsilon2 = 1.0e12
    while epsilon1 > threshold1:
        Y[:size1, :size2] = A.dot(X.dot(B))
        epsilon2 = 1.0e6
        while epsilon2 > threshold2:
            tmp = np.eye(size1, size1) / size1
            tmp += (one1.T.dot(Y.dot(one1)) / (size1 * size1)) * (np.eye(size1, size1))
            tmp -= Y / size1
            tmp = tmp.dot(one1.dot(one1.T))
            Y_new = Y + tmp - one1.dot(one1.T.dot(Y)) / size1
            Y_new = (Y_new + np.abs(Y_new)) / 2.0
            epsilon2 = np.abs(Y_new - Y).max()
            Y = Y_new
            # print("epsilon2 = %s" % epsilon2)

        print("epsilon2 = %s" % epsilon2)
        X_new = (1.0 - alpha) * X + alpha * Y[:size1, :size2]
        X_new = X_new / X_new.max()
        # X_new = X_new / X_new.sum(0)
        epsilon1 = np.abs(X_new - X).max()
        X = X_new
        print("epsilon1 = %s" % epsilon1)
        loss_X = np.linalg.norm(A - dot(X, dot(B, X.T)))
        print("Loss(X) = %s" % loss_X)
        
    return X


def fastPFP_faster(A, B, alpha=0.5, threshold1=1.0e-6, threshold2=1.0e-6, X=None, Y=None, verbose=True, max_iter1=100, max_iter2=100):
    """A faster and more efficient implementation of fastPFP().
    """
    size1 = A.shape[0]
    size2 = B.shape[0]
    if X is None: X = np.ones((size1, size2)) / (size1 * size2)
    if Y is None: Y = np.zeros((size1, size1))

    loss_best = epsilon1 = epsilon2 = 1.0e12
    iter1 = 0
    while epsilon1 > threshold1 and iter1 < max_iter1:
        Y[:size1, :size2] = A.dot(X.dot(B))
        epsilon2 = 1.0e6
        iter2 = 0
        while epsilon2 > threshold2 and iter2 < max_iter2:
            tmp = (1.0 + Y.sum() / size1 - Y.sum(1)) / size1  # Faster and requiring less memory
            Y_new = Y + tmp[:,None] - Y.sum(0) / size1
            Y_new = np.clip(Y_new, 0.0, 1.0e12)
            epsilon2 = np.abs(Y_new - Y).max()
            Y = Y_new
            iter2 += 1

        if verbose: print("epsilon2 = %s" % epsilon2)
        X_new = (1.0 - alpha) * X + alpha * Y[:size1, :size2]
        X_new = X_new / X_new.max()
        epsilon1 = np.abs(X_new - X).max()
        X = X_new
        if verbose: print("epsilon1 = %s" % epsilon1)
        iter1 += 1
        
    return X


def greedy_assignment(X):
    """A simple greedy algorithm for the assignment problem as
    proposed in the paper of fastPFP. If creates a proper partial
    permutation matrix (P) from the result (X) of the optimization
    algorithm fastPFP.
    """
    XX = X.copy()
    P = np.zeros(X.shape)
    while (XX > 0.0).any():
        row, col = np.unravel_index(XX.argmax(), XX.shape)
        P[row, col] = 1.0
        XX[row, :] = -1.0
        XX[:, col] = -1.0

    return P

