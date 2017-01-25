#!/usr/bin/env python
"""Bayesian linear regression using variational inference.

This version visualizes additional fits of the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical


def build_toy_dataset(N, noise_std=0.5):
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = 2.0 * X + 10 * np.random.normal(0, noise_std, size=N)
  X = X.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  return X, y


ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

# DATA
X_train, y_train = build_toy_dataset(N)
X_test, y_test = build_toy_dataset(N)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

# INFERENCE
T = 5000                        # Number of samples.
nburn = 100                     # Number of burn-in samples.
stride = 10                    # Frequency with which to plot samples.
qw = Empirical(params=tf.Variable(tf.random_normal([T, D])))
qb = Empirical(params=tf.Variable(tf.random_normal([T, 1])))

inference = ed.SGHMC({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(step_size=1e-3)


# CRITICISM

# Plot posterior samples.
sns.jointplot(qb.params.eval()[nburn:T:stride],
              qw.params.eval()[nburn:T:stride])
plt.show()

# Posterior predictive checks.
y_post = ed.copy(y, {w: qw.mean(), b: qb.mean()})
# This is equivalent to
# y_post = Normal(mu=ed.dot(X, qw.mean()) + qb.mean(), sigma=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Displaying prior predictive samples.")
n_prior_samples = 10

w_prior = w.sample(n_prior_samples).eval()
b_prior = b.sample(n_prior_samples).eval()

plt.scatter(X_train, y_train)

inputs = np.linspace(-1, 10, num=400, dtype=np.float32)
for ns in range(n_prior_samples):
    output = inputs * w_prior[ns] + b_prior[ns]
    plt.plot(inputs, output)

plt.show()

print("Displaying posterior predictive samples.")
n_posterior_samples = 10

w_post = qw.sample(n_posterior_samples).eval()
b_post = qb.sample(n_posterior_samples).eval()

plt.scatter(X_train, y_train)

inputs = np.linspace(-1, 10, num=400, dtype=np.float32)
for ns in range(n_posterior_samples):
    output = inputs * w_post[ns] + b_post[ns]
    plt.plot(inputs, output)

plt.show()
