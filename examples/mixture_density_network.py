#!/usr/bin/env python
"""
Mixture density network using maximum likelihood.

Probability model:
    Prior: None ("flat prior")
    Likelihood: Mixture sum of normals parameterized by a NN
Inference: Maximum a posteriori
"""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, merge
from sklearn.cross_validation import train_test_split

class MixtureDensityNetwork:
    """
    Mixture density network for outputs y on inputs x.

    p((x,y), z) = sum_{k=1}^K pi_k(x; z) Normal(y; mu_k(x; z), sigma_k(x; z))

    where pi, mu, sigma are the output of a neural network taking x
    as input and with parameters z.
    """
    def __init__(self, mixture_components):
        self.mixture_components = mixture_components

    def mapping(self, X):
        """pi, mu, sigma = NN(x; z)"""
        hidden1 = Dense(25, activation='relu')(X)  # fully-connected layer with 128 units and ReLU activation
        hidden2 = Dense(25, activation='relu')(hidden1)
        self.means = Dense(self.mixture_components)(hidden2)
        self.standard_deviations = Dense(self.mixture_components, activation=K.exp)(hidden2)
        self.weights = Dense(self.mixture_components, activation=K.softmax)(hidden2)

    def log_prob(self, xs, zs=None):
        """-1/M sum_{n=1}^M log p((x_n,y_n), z)"""
        # Note there are no parameters we're being Bayesian about. The
        # parameters z are baked into how we specify the neural
        # networks.
        X, y = xs
        self.mapping(X)
        result = tf.exp(norm.logpdf(y, self.means, self.standard_deviations))
        result = tf.mul(result, self.weights)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)

def build_toy_dataset():
    NSAMPLE = 6000
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_data = np.float32(np.random.normal(size=(NSAMPLE,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42)

ed.set_seed(42)

X_train, X_valid, y_train, y_valid = build_toy_dataset()
print(X_train.shape, X_valid.shape)
print(y_train.shape, y_valid.shape)

X = tf.placeholder(tf.float32, shape=(None, 1))
y = tf.placeholder(tf.float32, shape=(None, 1))

sess = tf.Session()
K.set_session(sess)

model = MixtureDensityNetwork(10)
loss = model.log_prob([X, y])
train = tf.train.AdamOptimizer().minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

NEPOCH = 20
train_loss = np.zeros(NEPOCH) # store the training progress here.
valid_loss = np.zeros(NEPOCH)
for i in range(NEPOCH):
    _, train_loss[i] = sess.run([train, loss],
                                feed_dict={X: X_train, y: y_train})
    valid_loss[i] = sess.run(loss, feed_dict={X: X_valid, y: y_valid})
    print(train_loss[i], valid_loss[i])
    pred_weights, pred_means, pred_std = sess.run(
        [model.weights, model.means, model.standard_deviations],
        feed_dict={X: X_valid})
