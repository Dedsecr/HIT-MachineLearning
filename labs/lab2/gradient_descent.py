import numpy as np
import matplotlib.pyplot as plt
from data import *
from utils import *

class gradient_descent:
    def __init__(self, dataset, lr=1e-1, epochs=70000, delta=1e-6, lambda_=1e-2):
        """
        梯度下降法初始化

        Args:
            dataset (tuple): 训练样本
            lr (float), optional): 学习率. Defaults to 1e-1.
            epochs (int, optional): 最大迭代次数. Defaults to 70000.
            delta (float), optional): 早停策略的参数值. Defaults to 1e-6.
            lambda_ (float), optional): 正则项系数. Defaults to 1e-2.
        """
        self.lambda_ = lambda_
        self.dataset = dataset
        self.X = transform(dataset[0])
        self.Y = dataset[1]
        self.lr = lr
        self.epochs = epochs
        self.delta = delta

    def gradient_descent(self):
        """
        梯度下降法

        Returns:
            tuple: (beta, 总迭代次数)
        """
        beta = np.random.normal(size=(self.X.shape[1], 1))
        lr_scheduler = lr_scheduler_MultiStep(self.epochs, self.lr)
        loss_last = loss = calc_loss(self.X, beta, self.Y, self.lambda_)
        last_epoch = self.epochs
        for epoch in range(self.epochs):
            beta = beta - self.lr * calc_gradient(self.X, beta, self.Y, self.lambda_)
            loss = calc_loss(self.X, beta, self.Y, self.lambda_)
            if early_stop(loss, loss_last, self.delta):
                last_epoch = epoch
                break
            else: loss_last = loss
            self.lr = lr_scheduler.step(epoch)
        return beta, last_epoch

# 画图
def draw(dataset, beta):
    X_0, X_1 = get_Xs(dataset)

    plt.plot(X_0[:, 0], X_0[:, 1], '.', color='r', label="$y=0$")
    plt.plot(X_1[:, 0], X_1[:, 1], '.', color='b', label="$y=1$")
    plot(beta, 0, 3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    loc, scale = get_data_option()
    size = 1000
    dataset = data_generator(size, loc, scale)
    beta, _ = gradient_descent(dataset).gradient_descent()
    draw(dataset, beta)
    print(calc_accuracy(dataset, beta))