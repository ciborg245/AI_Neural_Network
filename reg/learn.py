from matplotlib import pyplot as plt
from datasets import ds1

# (X, y) = ds1
#
# plt.scatter(x, y, c='red')
# plt.show()


(X, y) = ds1
#model deduction
theta = np.linalg.lstsq(X, y)[0]

#test data
plt.scatter(X[:, 1], y)
Xt = np.vstack(
    (
        np.ones(100),
        np.linspace(-10, 10, 100)
    )
).T

plt.plot(Xt[:, 1], np.matmul(Xt, theta), c='red')
plt.show()
