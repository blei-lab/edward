#!/usr/bin/env python
"""Logistic regression using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Empirical, Normal
from scipy.special import expit

ed.set_seed(123)
N = 5810  # number of data points
D = 54  # number of features

# DATA
w_true = np.random.randn(D)
X_data = np.random.randn(N, D)
p = expit(np.dot(X_data, w_true))
y_data = np.array([np.random.binomial(1, i) for i in p])

# MODEL
X = tf.Variable(X_data.astype(np.float32), trainable=False)
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
y = Bernoulli(logits=ed.dot(X, w))

# INFERENCE
T = 5000
qw = Empirical(params=tf.Variable(tf.zeros([T, D])))
inference = ed.HMC({w: qw}, data={y: y_data})
inference.run(step_size=0.05)

# CRITICISM
print("Mean squared error in true values to inferred posterior mean:")
print(tf.reduce_mean(tf.square(w_true - qw.mean())).eval())
