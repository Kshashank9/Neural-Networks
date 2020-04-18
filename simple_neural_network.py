import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_der(x):
    return x * (1 - x)


training_set = np.array([[0, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [0, 1, 1]])

output_set = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
w_btw_nodes = 2 * np.random.random((3, 1)) - 1

for i in range(50000):
    input_layer = training_set
    outputs = sigmoid(np.dot(input_layer, w_btw_nodes))

    error = output_set - outputs
    updating_net = error * sig_der(outputs)
    w_btw_nodes += np.dot(input_layer.T, updating_net)


print("Trained Output: ")
print(w_btw_nodes)

print("Outputs: ")
print(outputs)


