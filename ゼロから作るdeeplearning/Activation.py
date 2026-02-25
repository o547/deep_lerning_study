import numpy as np


class Activation:
    def __init__(self):
        import numpy as np

    def stepup(self, x):
        return np.array(x > 0, dtype=int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, X):
        c = np.max(X)
        exp_sum = np.sum(np.exp(X - c))
        return np.exp(X - c) / exp_sum


import numpy as np

act = Activation()
x = np.array([1010, 1000, 990])
print(act.softmax(x))
