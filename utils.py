import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math


#normalizing both training and testing
def normalization(x, xt):
    mn = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    x = (x - mn) / std
    xt = (xt - mn) / std
    return [x, xt]

def read(filename):
    df = pd.read_csv(filename)
    return df

#function to preprocess the data
#unwrap the return value into 4 variables
#Eg: x_train, y_train, x_test, y_test = preprocessing('data.csv')
def preprocessing(df):
    training = df.sample(frac=0.8, random_state = 79)
    testing = df.drop(training.index)
    target = df.columns[-1]

    x_train = np.array(training.loc[:, df.columns != target], dtype = np.float64)
    y_train = np.array(training.loc[:, [target]], dtype = np.float64)
    x_test = np.array(testing.loc[:, df.columns != target], dtype = np.float64)
    y_test = np.array(testing.loc[:, [target]], dtype = np.float64)

    x_train, x_test = normalization(x_train, x_test)
    ones = np.ones(len(x_train))
    ones_test = np.ones(len(x_test))
    ones = ones.reshape(len(ones), 1)
    ones_test = ones_test.reshape(len(ones_test), 1)
    x_train = np.hstack([ones, x_train])
    x_test = np.hstack([ones_test, x_test])
    return [x_train, y_train, x_test, y_test]


#cost function and gradient (in the full model)
def costFunction(X, W, Y):
    difference = (X@W) - Y
    ans = (difference.T@difference) / (2)
    return ans

def gradientDescent(X, Y, W, alpha, iter):
    cost = []
    for i in range(iter):
        difference = (X@W) - Y
        gradient = (X.T@difference) 
        W -= alpha * gradient
        cost.append(costFunction(X, W, Y)[0])
    return cost, W

#unwrap the return value into two variables
#Eg: costs, weights = model(X_train, Y_train)
def model(X_train, Y_train,iters,alpha):
    weights = [0 for i in range(len(X_train[0]))]
    weights = np.array(weights, dtype = np.float64)
    weights = weights.reshape(len(weights), 1)
    costs, weights = gradientDescent(X_train, Y_train, weights, alpha, iters)
    return [costs, weights]


def rootMeanSquareError(y_test, y_predicted):
    error = 0
    for i in range(len(y_predicted)):
        error += (y_predicted[i][0] - y_test[i][0]) ** 2
    error /= len(y_predicted)
    error = math.sqrt(error)
    return error

#unwrap the return value into two variables
#Eg: error_test, error_train = predictions(x_train, y_train, x_test, y_test, w)
def predictions(x_train, y_train, x_test, y_test, w):
    y_predicted_test = x_test@w
    error_test = rootMeanSquareError(y_test, y_predicted_test)
    y_predicted_train = x_train@w
    error_train = rootMeanSquareError(y_train, y_predicted_train)
    return [error_test, error_train]

def regression(df,iters,alpha):
    X_train, Y_train, X_test, Y_test = preprocessing(df)
    cost, weight = model(X_train, Y_train, iters, alpha)
    error_test, error_train = predictions(X_train, Y_train, X_test, Y_test, weight)
    return error_test, error_train
