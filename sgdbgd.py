import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

#批量梯度下降法
def BGD(X, Y, theta, alpha, maxIterations):
    m, n = np.shape(X)
    added_X = np.ones((m, n+1))
    added_X[:,1:] = X
    Y = Y.reshape(m,1)
    for i in range(0, maxIterations):
        gradient = np.dot(added_X.transpose(), (np.dot(added_X, theta) - Y)) / m
        # print(gradient)
        theta = theta - alpha * gradient
    return theta

#随机梯度下降法
def SGD(X, Y, theta, alpha, maxIterations):
    m, n = np.shape(X)
    data = []
    for i in range(m):
        data.append(i)
    added_X = np.ones((m, n + 1))
    added_X[:, 1:] = X
    Y = Y.reshape(m, 1)

    for i in range(0,maxIterations):
        H = np.dot(added_X, theta)
        loss = H - Y
        index = random.randint(0,m-1)
        gradient = loss[index]*added_X[index]
        theta = theta - alpha * gradient.reshape(theta.shape[0],1)
    return theta

def predict(X, theta):
    m, n = np.shape(X)
    x_test = np.ones((m, n+1))
    x_test[:, 1:] = X
    res = np.dot(x_test, theta)
    return res

def featureNormalize(X):
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
        mu[0, i] = np.mean(X[:, i])  # 均值
        sigma[0, i] = np.std(X[:, i])  # 标准差
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


allData = pd.read_csv('Advertising.csv')
m, n = np.shape(allData)
print(m)
print(n)
ratio = 0.7
X = allData.values[:, :-1].reshape((-1, 3))
labels = allData.values[:, -1].reshape((-1, 1))

x, mu, sigma = featureNormalize(X)

m,n = np.shape(x)
print(m)
print(n)
theta = np.ones(n+1).reshape(n+1,1)
alpha = 0.01
maxIteration = 10000

theta_BGD = BGD(x, labels, theta, alpha, maxIteration)
print(theta_BGD)
print("==============================================")
theta_SGD = SGD(x, labels, theta, alpha, maxIteration)
print(theta_SGD)
