import network
from math import sqrt, trunc
import numpy as np
from PIL import Image


layer2Len = 45
layer3Len = 10
# layer4Len = 10

H1 = np.array([np.random.uniform(0, 0.15, 784) for j in range(layer2Len)])
H2 = np.array([np.random.uniform(0, 0.15, layer2Len) for j in range(layer3Len)])
# B2 = np.ones(layer2Len)
# H3 = np.array([np.random.uniform(0, 0.001, layer3Len) for j in range(layer4Len)])
# B3 = np.ones(layer3Len)
weights = [H1, H2]
# biases = [B2, B3]

circle = np.load('data/full_numpy_bitmap_circle.npy')[:100000]
face = np.load('data/full_numpy_bitmap_face.npy')[:100000]
house = np.load('data/full_numpy_bitmap_house.npy')[:100000]
square = np.load('data/full_numpy_bitmap_square.npy')[:100000]
tree = np.load('data/full_numpy_bitmap_tree.npy')[:100000]
triangle = np.load('data/full_numpy_bitmap_triangle.npy')[:100000]

# mickey = np.load('data/full_numpy_bitmap_panda.npy')[:5000]

# circle = np.load('data/full_numpy_bitmap_circle.npy')
# face = np.load('data/full_numpy_bitmap_face.npy')
# house = np.load('data/full_numpy_bitmap_house.npy')
# square = np.load('data/full_numpy_bitmap_square.npy')
# tree = np.load('data/full_numpy_bitmap_tree.npy')
# triangle = np.load('data/full_numpy_bitmap_triangle.npy')
# mickey = np.load('data/full_numpy_bitmap_panda.npy')
mickey = np.load('data/mickeymouse.npy')
egg = np.load('data/egg.npy')
question = np.load('data/questionmark.npy')
sadface = np.load('data/sadface.npy')

data = np.concatenate((circle, face, house, square, tree, triangle, mickey, egg, question, sadface))
data[data > 1] = 1
print(data.shape)
# for i in range(28):
#     print([data[450000][i*28 + j] for j in range(28)])
results = np.concatenate((
    np.repeat(0, len(circle)),
    np.repeat(1, len(face)),
    np.repeat(2, len(house)),
    np.repeat(3, len(square)),
    np.repeat(4, len(tree)),
    np.repeat(5, len(triangle)),
    np.repeat(6, len(mickey)),
    np.repeat(7, len(egg)),
    np.repeat(8, len(question)),
    np.repeat(9, len(sadface)),
    ))

dataset = list(map(lambda x, y: (x, y), data, results))
np.random.shuffle(dataset)

trainingLen = trunc(len(dataset) * 0.7)
# for j in range(2):
for i in range(0, trainingLen, 50):
    batch = dataset[i:i+50]
    delta = network.back_propagation(batch, weights)
    # weights = weights - (6.0 / 50)*delta
    weights = weights - (0.1)*delta
    print(i)


testLen = trunc(len(dataset) * 0.85)
test = dataset[testLen:]
cont = 0
for i in test:
    if i[1] == np.argmax(network.feed_forward(np.array([i[0]]), weights)[2]):
        cont += 1

# print('rate {}'.format(j/10))
print('{0} / {1}'.format(cont, len(test)))


np.save('weights', weights)

file = Image.open("input.bmp")
input = np.array(file)
inputArr = [0 if input[i][j][0] == 255 else 1 for i in range(28) for j in range(28)]
# for i in range(28):
#     print([inputArr[i*28 + j] for j in range(28)])
# print(inputArr)

res = network.feed_forward(np.array([inputArr]), weights)
print(res[2])
resLabels = ['circle', 'face', 'house', 'square', 'tree', 'triangle', 'mickey', 'egg', 'question', 'sadface']
