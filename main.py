import network
import random
import numpy as np

# x = np.array([[[1], [2]], [[1], [2]], [[1], [2]]])
# x = np.array([[1, 2], [1, 2], [1, 2]])



# x = np.array([[1], [2]])
# H2 = np.array([
#     [0.1, 0.3, 0.5],
#     [0.2, 0.4, 0.6]
#     ])
# H3 = np.array([0.7, 0.8, 0.9])
#
# res = network.feed_forward(x, H2, H3)
# print(res)


layer2Len = 250
layer3Len = 250
# H2 = np.array([[random.uniform(0, 1) for i in range(784)] for j in range(layer2Len)])
# H3 = np.array([[random.uniform(0, 1) for i in range(layer2Len)] for j in range(layer3Len)])
H2 = np.array([[random.uniform(0, 1) for i in range(784)] for j in range(layer2Len)])
H3 = np.array([[random.uniform(0, 1) for i in range(layer2Len)] for j in range(layer3Len)])
# print(H2)
# print(H2.shape)

a = np.load('data/full_numpy_bitmap_circle.npy')
a = a[:5000]
# print(len(a))
print(network.feed_forward(a, H2, H3))
# print(len(a[0]))
# print(len(a))

# for i in range(28):
#     print([a[0][i*28+j] for j in range(28)])
