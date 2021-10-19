import numpy as np
import matplotlib.pyplot as plt
from utils import *


def data_generator(size=(100, 150), loc=np.array([[1, 3],[2, 5]]), cov=[[0.4, 0.2], [0.2, 0.4]]):
    def generate(siz):
        X = np.zeros((siz * 2, 2))
        Y = np.zeros((siz * 2, 1))
        X[:siz, :] = np.random.multivariate_normal(loc[0], cov, size=siz)
        X[siz:, :] = np.random.multivariate_normal(loc[1], cov, size=siz)
        Y[siz:] = 1
        return X, Y
    train_data, test_data = generate(size[0]), generate(size[1])
    return train_data, test_data

'''
def data_generator(size=(100, 150), loc=np.array([[1, 3],[2, 5]]), scale=[0.4 * np.ones(2), 0.4 * np.ones(2)], concatenate=True):
    def generate(siz):
        X = np.empty((2, siz, loc[0].size))
        Y = np.array([np.zeros((siz, 1)), np.ones((siz, 1))])

        X[0] = np.random.normal(loc[0], scale[0], (siz, loc[0].size))
        X[1] = np.random.normal(loc[1], scale[1], (siz, loc[0].size))

        if concatenate:
            X = np.concatenate((X[0], X[1]), axis=0)
            Y = np.concatenate((Y[0], Y[1]), axis=0)
        return X, Y
    train_data, test_data = generate(size[0]), generate(size[1])
    return train_data, test_data
'''

def show(dataset):
    plt.figure(figsize=(8, 8))
    messages = ['with Bayes Hypothesis', 'without Bayes Hypothesis']
    titles = ['Train Data', 'Test Data']
    for i in range(2):
        for j in range(2):
            X_0, X_1 = get_Xs(dataset[i][j])
            plt.subplot(2, 2, i * 2 + j + 1)
            plt.plot(X_0[:, 0], X_0[:, 1], '.', color='r', label="$y=0$")
            plt.plot(X_1[:, 0], X_1[:, 1], '.', color='b', label="$y=1$")
            plt.title('{} {}'.format(titles[j], messages[i]))
            plt.legend()
    plt.show()

def get_data_option(loc1=(1, 3), loc2=(2, 5), scale=0.2, conv=0.1):
    loc = np.array([[loc1[0], loc1[1]],[loc2[0], loc2[1]]])
    cov = [[scale, conv], [conv, scale]]
    return loc, cov

if __name__ == '__main__':
    size = (100, 150)
    loc, cov = get_data_option()
    dataset0 = data_generator(size, loc, cov)
    loc, cov = get_data_option(conv=0)
    dataset1 = data_generator(size, loc, cov)
    show((dataset0, dataset1))