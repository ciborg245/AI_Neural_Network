y = array(10)

y = array([4, 7, 1])

y = np.linspace(0, 5, 100)

b = np.array([[1, 2], [3, 4], [5, 6]])
b.shape --> (3, 2)
b.reshape(3, 2) --> [[1, 2, 3], [4, 5, 6]]

a = np.linspace(0, 99, 100)
a.reshape(10, 10)


b.T --> transpuesta
b.transpose()

ones = np.ones((10, 1))
zeros = np.zeros((10, 1))
np.hstack(ones, zeros)
np.vstack(ones, zeros)
