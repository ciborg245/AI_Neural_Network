import numpy as np
import math

def feed_forward(X, H1, H2):
    sig = numpy.vectorize(sig)

    X = np.vstack(
        (
            np.array([0]),
            X
        )
    )

    Z2 = np.matmul(H1, X)

    A2 = sigmoide(Z2)
    A2 = np.vstack(
        (
            np.array([0]),
            A2
        )
    )
    Z3 = np.matmul(H2, A2)

    A3 = sigmoide(Z3)

    return A3

def sig(vec):
    return 1 / (1 + math.exp(-x))
