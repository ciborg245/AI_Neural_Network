import numpy as np
import matplotlib.pyplot as plt

#Nuevas!
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

x, y = datasets.load_iris(return_X_y = True)
Y = y == 0
X = x[:, 0:2]

X = np.hstack((
    X[:, 0],
    X[:, 1],
    X[:, 0] ** 2,
    X[:, 0] ** 3,
    X[:, 1] ** 2,
    X[:, 1] ** 3,
    X[:, 0] * X[:, 1]
    X[:, 0] ** 2 * X[:, 1]
    X[:, 0] * X[:, 1] ** 2
    X[:, 0] ** 2 * X[:, 1] ** 2
))

x0_min = min(X[:, 0])
x1_min = min(X[:, 1])
x0_max = max(X[:, 0])
x1_max = max(X[:, 1])

step = 0.02
x0test, x1test = np.meshgrid(
    np.arange(x0_min, x0_max, step),
    np.arange(x1_min, x1_max, step)
)

X_test = np.c_[
    x0test.ravel(),
    x1test.ravel()
]

logreg = LogisticRegression(
    C=10000,
    solver='lbfgs',
    multi_class='multinomial'
)

# Training
logreg.fit(X, Y)

#Test
H = logreg.predict(X_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c = H, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
