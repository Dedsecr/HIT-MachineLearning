import numpy as np
import matplotlib.pyplot as plt
from data import *

def transform(X):
    X_transformed = np.zeros((X.shape[0], X.shape[1] + 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_transformed[i][j] = X[i][j]
        X_transformed[i][X.shape[1]] = 1
    return X_transformed

def calc_loss(X, beta, Y, lambda_):
    """
    计算loss

    Args:
        X (list): X
        beta (list): beta
        Y (list): Y
        lambda_ (float | None): 正则项系数

    Returns:
        float: loss
    """
    loss = np.mean(-Y.T @ X @ beta + np.log(1 + np.exp(X @ beta)))
    if lambda_ != None:
        loss += lambda_ / 2 * np.mean(beta.T @ beta)
    return loss

def calc_gradient(X, beta, Y, lambda_):
    """
    计算梯度

    Args:
        X (list): X
        beta (list): beta
        Y (list): Y
        lambda_ (float | None): 正则项系数

    Returns:
        list: 梯度
    """
    exp = np.exp(X @ beta)
    gradient = (X.T @ (exp / (1 + exp)) - X.T @ Y) / X.shape[0]

    if lambda_ != None:
        gradient += lambda_ * beta
    return gradient

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def model(X, beta):
    output = sigmoid(X @ beta)
    return output

def calc_accuracy(dataset, beta):
    X, Y = transform(dataset[0]), dataset[1].flatten()
    output = model(X, beta)
    output = np.array([1 if x >= 0.5 else 0 for x in output])
    output = np.logical_xor(output, Y)
    accuracy = (len(Y) - np.sum(output, axis=0)) / len(Y)
    return accuracy

def early_stop(loss, loss_last, delta):
    """
    早停策略

    Args:
        loss (float): 当前的loss
        loss_last (float): 上一个loss
        delta (float): 早停策略参数

    Returns:
        bool: 是否停止
    """
    return abs(loss - loss_last) < delta


class lr_scheduler_MultiStep:
    def __init__(self, epochs, lr):
        """
        学习率下降策略

        Args:
            epochs (int): 总迭代轮数
            lr (float): 初始学习率
        """
        self.epochs = epochs
        self.lr = lr
        self.gamma = 0.1
        self.milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    def step(self, epoch):
        """
        更新学习率

        Args:
            epoch (int): 当前迭代轮数

        Returns:
            float: 新的学习率
        """
        if epoch in self.milestones:
            self.lr *= self.gamma
        return self.lr

def get_Xs(dataset):
    X, Y = dataset
    X_0, X_1 = [], []
    for index in range(Y.shape[0]):
        if Y[index][0] == 0: X_0.append(X[index])
        else: X_1.append(X[index])
    return np.array(X_0), np.array(X_1)

def plot(beta, start, end):
    x0 = np.linspace(start, end, 1000)
    x1 = -(beta[0] * x0 + beta[2]) / beta[1]
    plt.plot(x0, x1)