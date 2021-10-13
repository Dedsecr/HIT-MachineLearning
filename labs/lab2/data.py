import numpy as np
import matplotlib.pyplot as plt


def data_generator(size=100, loc=np.array([[1, 3],[2, 5]]), scale=[0.5 * np.ones(2), 0.5 * np.ones(2)], concatenate=True):
    X = np.empty((2, size, loc[0].size))
    Y = np.array([np.zeros((size, 1)), np.ones((size, 1))])

    X[0] = np.random.normal(loc[0], scale[0], (size, loc[0].size))
    X[1] = np.random.normal(loc[1], scale[1], (size, loc[0].size))

    if concatenate:
        X = np.concatenate((X[0], X[1]), axis=0)
        Y = np.concatenate((Y[0], Y[1]), axis=0)
    return X, Y

def show(dataset):
    X, _ = dataset
    plt.plot(X[0][:, 0], X[0][:, 1], '.', color='r', label="$y=0$")
    plt.plot(X[1][:, 0], X[1][:, 1], '.', color='b', label="$y=1$")
    plt.legend()
    plt.show()

def get_data_option(loc1=(1, 3), loc2=(2, 5), scale=0.4):
    loc = np.array([[loc1[0], loc1[1]],[loc2[0], loc2[1]]])
    scale = [scale * np.ones(2), scale * np.ones(2)]
    return loc, scale

if __name__ == '__main__':
    loc, scale = get_data_option()
    size = 100
    dataset = data_generator(size, loc, scale, False)
    show(dataset)