def linear_cost(o, x, y):
    r1 = sum(
        y[i] -
        o[j]*sum([y[i] - x[i][j] for i in range(y)])
        for j in range(o)
    )

def linear_cost(theta, X, y):
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    return sq.sum()
