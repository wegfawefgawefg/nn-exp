import math
import numpy as np
import matplotlib.pyplot as plt

#def sigmoid(x):
#    return 1 / (1 + math.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    # Test sigmoid
    x = np.arange(-10, 10, 0.1)
    y = [sigmoid(i) for i in x]
    plt.plot(x, y)
    plt.show()
