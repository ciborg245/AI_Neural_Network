import numpy as np
import math

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

sigmoide = np.vectorize(sig)

def feed_forward(x, theta):
    thetaLen = len(theta)

    x = x.T
    Z = []
    A = [x]
    for i in range(thetaLen):
        # print(A[i].shape)
        if i > 0:
            m, n = theta[i].shape
            # print('theta')
            # print(theta[i].shape)

            theta_mod = np.hstack((
                np.ones(m).reshape(m, 1),
                # np.random.uniform(0, 0.1, m).reshape(m, 1),
                theta[i]
            ))
            # print('theta mod')
            # print(theta_mod.shape)
        else:
            theta_mod = np.array(theta[i])
        z_calc = theta_mod @ A[i]
        Z.append(z_calc)
        # print('z calculated')

        a_calc = sigmoide(Z[i])
        # print('sigmoid calculated')

        if i < thetaLen - 1:
            m, n = a_calc.shape
            # print(a_calc.shape)
            a_calc = np.vstack((
                np.ones(n),
                a_calc
            ))
        A.append(a_calc)

    for i in range(1, len(A)-1):
        A[i] = A[i][1:]
    # print(A[1].shape)
    return np.array(A)

def getY(index):
    arr = np.zeros(10)
    arr[index] = 1.0
    return arr

def back_propagation(input, theta):
    m = len(theta)
    x = np.array([el[0] for el in input])
    # print(x)
    # print(input)
    A = feed_forward(x, theta)

    delta = np.array([np.zeros(theta[i].shape) for i in range(m)])
    # print(delta[0].shape)
    # print(delta[1].shape)
    # y = [
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # ]
    for n in range(len(x)):
    # for n in range(1):
        error = []
        for i in range(1, m+1):
            if i == 1:
                error.append(
                    np.array(A[m-i+1].T[n] - getY(input[i][1]))
                )

            else:
                error.append(calculate_error(theta[m-i+1], error[i-2], A[m-i+1].T[n]))
            # print(np.array([error[i-1]]).T.shape)
            # print(np.array([A[m-i].T[n]]).shape)
            # print((np.array([error[i-1]]).T @ np.array([A[m-i+1].T[n]])).shape)
            delta[m-i] = delta[m-i] + (np.array([error[i-1]]).T @ np.array([A[m-i].T[n]]))

    # print(error[0].shape)
    # print(error[1].shape)
    # print(A[0].T[0])
    # print(delta[0][0])

    return delta

def calculate_error(theta, sub_error, A):
    # print((theta.T @ sub_error) * (A * (np.ones(len(A))-A)))
    # print((theta.T @ sub_error) * (A * (np.ones(len(A))-A)))
    return np.array((theta.T @ sub_error) * (A * (np.ones(len(A))-A)))
