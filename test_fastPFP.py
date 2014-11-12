import numpy as np
from create_dataset import create_dataset_artificial
import matplotlib.pyplot as plt
from fastPFP import fastPFP_faster, fastPFP

if __name__ == '__main__':

    np.random.seed(0)

    size1 = 35
    size2 = 30

    same = True

    print("Simple 2D example of fastPFP.")
    print("Same: %s" % same)
    A, B, X1, X2 = create_dataset_artificial(size1, size2, same)

    print("fastPFP:")
    X = fastPFP_faster(A, B, alpha=0.5, threshold1=1.0e-4, threshold2=1.0e-4)
    P = (X == X.max(1)[:,None])
    loss_X = np.linalg.norm(A - X.dot(B.dot(X.T)))
    loss_P = np.linalg.norm(A - P.dot(B.dot(P.T)))
    print("Loss(X) = %s" % loss_X)
    print("Loss(P) = %s" % loss_P)

    print("")
    print("Plotting.")
    plt.figure()
    X2 = X2 + np.array([1.0, 0.5]) # Adding some constant displacement for visualization purpose.
    plt.plot(X1[:,0], X1[:,1], 'ro', markersize=10)
    plt.plot(X2[:,0], X2[:,1], '*b', markersize=10)
    mapping12 = P.argmax(1)
    if size2 >= size1:
        for i in range(size1):
            # plt.plot([X1[i,0], X2[mapping12[i],0]], [X1[i,1], X2[mapping12[i],1]], 'r-')
            temp = X2[mapping12[i]] - X1[i]
            plt.arrow(X1[i, 0], X1[i, 1], temp[0], temp[1], head_width=0.05, head_length=0.05, length_includes_head=True)

    else:
        mapping21 = P.argmax(0)
        for i in range(size2):
            temp = X1[mapping21[i]] - X2[i]
            plt.arrow(X2[i, 0], X2[i, 1], temp[0], temp[1], head_width=0.05, head_length=0.05, length_includes_head=True)

    plt.show()
