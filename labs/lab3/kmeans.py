import numpy as np
from data import *
from utils import *
import collections


def kmeans(X, k, epochs=10000, epsilon=1e-7):
    c = X[np.random.choice(range(X.shape[0]), k)]
    for epoch in range(epochs):
        tags = np.zeros((X.shape[0]), dtype=np.int)
        for i in range(X.shape[0]):
            _distances = [np.linalg.norm(x - X[i], 2) for x in c]
            tags[i] = _distances.index(min(_distances))
        new_c = np.zeros((k, X.shape[1]))
        for i in range(X.shape[0]):
            new_c[tags[i]] += X[i]
        counter = collections.Counter(tags)
        # if 0 in [counter[tag] for tag in range(k)]:
        #     raise ValueError("")
        new_c /= np.array([counter[tag] for tag in range(k)]).reshape((k, 1))
        if np.linalg.norm(new_c - c, 2) < epsilon:
            c = new_c
            break
        print(np.linalg.norm(new_c - c, 2))
        c = new_c
        # print(c)
    return c, epoch

# 画图
def show(dataset, c, epoch):
    X, Y = dataset
    k = max(y for y in Y) + 1
    colors = ['r', 'g', 'y', 'b']
    for i in range(k):
        ix = np.where(Y == i)
        plt.plot(X[ix, 0], X[ix, 1], '.', color=colors[i])
    plt.scatter(c[:, 0], c[:, 1], c='b', marker="x", s=250)
    plt.title('epoch={}'.format(epoch))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    k, size, loc, cov = get_data_option(size=200)
    dataset = data_generator(k, size, loc, cov)
    c, epoch = kmeans(dataset[0], k)
    show(dataset, c, epoch)