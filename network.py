import numpy as np
import math

def sig(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(x, H1, H2):
    sigmoide = np.vectorize(sig)

    m, n = x.shape
    # print(x)
    # print(np.ones(m))
    # print(np.ones(n).reshape(n, 1))
    A1 = np.hstack((
        np.ones(m).reshape(m, 1),
        x
    ))

    print(A1)

    # print(x)
    # print(H1)
    # a1 = np.vstack((1, x))
    #
    print(H1.shape)
    # z2 = np.matmul(H1, a1)
    #
    # a2 = sigmoide(z2)
    # a2 = np.vstack((1, a2))
    #
    # z3 = np.matmul(H2, a2)
    #
    # a3 = sigmoide(z3)
    #
    # return a3
