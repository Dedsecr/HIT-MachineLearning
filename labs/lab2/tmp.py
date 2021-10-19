import numpy as np

def data_generator(size=5, loc=np.array([[1, 3],[2, 5]]), scale=[0.5 * np.ones(2), 0.5 * np.ones(2)], concatenate=True):
    X = np.empty((2, size, loc[0].size))
    Y = np.array([np.zeros((size, 1)), np.ones((size, 1))])

    print(X.shape, Y.shape)

    X[0] = np.random.normal(loc[0], scale[0], (size, loc[0].size))
    X[1] = np.random.normal(loc[1], scale[1], (size, loc[0].size))

    print(X)

    if concatenate:
        X = np.concatenate((X[0], X[1]), axis=0)
        Y = np.concatenate((Y[0], Y[1]), axis=0)
    return X, Y


if __name__ == "__main__":
    data_generator()