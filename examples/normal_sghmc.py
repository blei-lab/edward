#!/usr/bin/env python
"""Correlated normal posterior. Inference with stochastic gradient Hamiltonian
Monte Carlo.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from edward.models import Empirical, MultivariateNormalFull

plt.style.use("ggplot")

# Plotting helper function.


def mvn_plot_contours(z, label=False, ax=None):
  """
  Plot the contours of 2-d Normal or MultivariateNormalFull object.
  Scale the axes to show 3 standard deviations.
  """
  sess = ed.get_session()
  mu = sess.run(z.mu)
  mu_x, mu_y = mu
  Sigma = sess.run(z.sigma)
  sigma_x, sigma_y = np.sqrt(Sigma[0, 0]), np.sqrt(Sigma[1, 1])
  xmin, xmax = mu_x - 3 * sigma_x, mu_x + 3 * sigma_x
  ymin, ymax = mu_y - 3 * sigma_y, mu_y + 3 * sigma_y
  xs = np.linspace(xmin, xmax, num=100)
  ys = np.linspace(ymin, ymax, num=100)
  X, Y = np.meshgrid(xs, ys)
  T = tf.convert_to_tensor(np.c_[X.flatten(), Y.flatten()], dtype=tf.float32)
  Z = sess.run(tf.exp(z.log_prob(T))).reshape((len(xs), len(ys)))
  if ax is None:
    fig, ax = plt.subplots()
  cs = ax.contour(X, Y, Z)
  if label:
    plt.clabel(cs, inline=1, fontsize=10)


# Example body.
ed.set_seed(42)

# MODEL
z = MultivariateNormalFull(mu=tf.ones(2),
                           sigma=tf.constant([[1.0, 0.8], [0.8, 1.0]]))

# INFERENCE
qz = Empirical(params=tf.Variable(tf.random_normal([5000, 2])))

inference = ed.SGHMC({z: qz})
inference.run(step_size=0.02)

# CRITICISM
sess = ed.get_session()
mean, std = sess.run([qz.mean(), qz.std()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior std:")
print(std)

# VISUALIZATION
fig, ax = plt.subplots()
trace = sess.run(qz.params)
ax.scatter(trace[:, 0], trace[:, 1], marker=".")
mvn_plot_contours(z, ax=ax)
plt.show()
