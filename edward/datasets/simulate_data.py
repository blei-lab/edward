#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from edward.stats import norm
from sklearn.cross_validation import train_test_split

def simulate_regression_data(N=40, noise_std=0.1, coeff=None, n_dims = 1, transform=None):
    if coeff == None and n_dims ==1:
        x  = np.concatenate([np.linspace(0, 2, num=N/2),
                             np.linspace(6, 8, num=N/2)])
        if transform == 'cosine':
            y = np.cos(x) + norm.rvs(0, noise_std, size=N)
        else:
            y = 0.075*x + norm.rvs(0, noise_std, size=N)
        x = (x - 4.0) / 4.0
        x = x.reshape((N, 1))
    elif coeff == None:
        x = np.random.randn(N, n_dims).astype(np.float32)
        y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=N)
    else:
        n_dims = len(coeff)
        x = np.random.randn(N, n_dims).astype(np.float32)
        y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=N)
    return {'x': x, 'y': y}

def simulate_binary_classification_data(N=40, D=1, noise_std=0.1):
    x  = np.linspace(-3, 3, num=N)
    y = np.tanh(x) + norm.rvs(0, noise_std, size=N)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    x = (x - 4.0) / 4.0
    x = x.reshape((N, D))
    return {'x': x, 'y': y}


def simulate_mdn_data(N=6000,test_split=True):
    x = tf.placeholder(tf.float32, shape=(None, 1))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    if not test_split:
        return {'X':x, 'y': y}
    else:
        y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, N))).T
        r_data = np.float32(np.random.normal(size=(N, 1))) # random noise
        x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42)
        return {'X':x, 'y': y}, X_train, X_test, y_train, y_test


