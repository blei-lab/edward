#!/usr/bin/env python
"""
Mixture density network using maximum likelihood.

Probability model:
    Prior: None ("flat prior")
    Likelihood: Mixture sum of normals parameterized by a NN
Inference: Maximum a posteriori
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.stats import norm
from keras import backend as K
from keras.layers import Dense
from edward.datasets import simulate_mdn_data


class MixtureDensityNetwork:
    """
    Mixture density network for outputs y on inputs x.

    p((x,y), (z,theta))
    = sum_{k=1}^K pi_k(x; theta) Normal(y; mu_k(x; theta), sigma_k(x; theta))

    where pi, mu, sigma are the output of a neural network taking x
    as input and with parameters theta. There are no latent variables
    z, which are hidden variables we aim to be Bayesian about.
    """
    def __init__(self, K):
        self.K = K

    def neural_network(self, X):
        """pi, mu, sigma = NN(x; theta)"""
        hidden1 = Dense(25, activation='relu')(X)  # fully-connected layer with 25 hidden units
        hidden2 = Dense(25, activation='relu')(hidden1)
        self.mus = Dense(self.K)(hidden2)
        self.sigmas = Dense(self.K, activation=K.exp)(hidden2)
        self.pi = Dense(self.K, activation=K.softmax)(hidden2)

    def log_prob(self, xs, zs=None):
        """log p((xs,ys), (z,theta)) = sum_{n=1}^N log p((xs[n,:],ys[n]), theta)"""
        # Note there are no parameters we're being Bayesian about. The
        # parameters are baked into how we specify the neural networks.
        X, y = xs['X'], xs['y']
        self.neural_network(X)
        result = tf.exp(norm.logpdf(y, self.mus, self.sigmas))
        result = tf.mul(result, self.pi)
        result = tf.reduce_sum(result, 1)
        result = tf.log(result)
        return tf.reduce_sum(result)


ed.set_seed(42)
model = MixtureDensityNetwork(10)
    
data, X_train, X_test, y_train, y_test = simulate_mdn_data()
print("Size of features in training data: {:s}".format(X_train.shape))
print("Size of output in training data: {:s}".format(y_train.shape))
print("Size of features in test data: {:s}".format(X_test.shape))
print("Size of output in test data: {:s}".format(y_test.shape))
    
inference = ed.MAP(model, data)
sess = ed.get_session()
K.set_session(sess)
inference.initialize()

NEPOCH = 20
train_loss = np.zeros(NEPOCH)
test_loss = np.zeros(NEPOCH)
for i in range(NEPOCH):
    _, train_loss[i] = sess.run([inference.train, inference.loss],
                                feed_dict={data['X']: X_train, data['y']: y_train})
    test_loss[i] = sess.run(inference.loss, feed_dict={data['X']: X_test, data['y']: y_test})
    print("Train Loss: {:0.3f}, Test Loss: {:0.3f}".format(train_loss[i], test_loss[i]))

pred_weights, pred_means, pred_std = sess.run(
        [model.pi, model.mus, model.sigmas], feed_dict={data['X']: X_test})
