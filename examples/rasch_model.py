#!/usr/bin/env python
"""Rasch model (Rasch, 1960)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Empirical
from scipy.special import expit

# DATA
nsubj = 200
nitem = 25
trait_true = np.random.normal(size=[nsubj, 1])
thresh_true = np.random.normal(size=[1, nitem])
X_data = np.random.binomial(1, expit(trait_true - thresh_true))

# MODEL
trait = Normal(loc=tf.zeros([nsubj, 1]), scale=tf.ones([nsubj, 1]))
thresh = Normal(loc=tf.zeros([1, nitem]), scale=tf.ones([1, nitem]))
X = Bernoulli(logits=trait - thresh)

# INFERENCE
T = 5000  # number of posterior samples
q_trait = Empirical(params=tf.Variable(tf.zeros([T, nsubj, 1])))
q_thresh = Empirical(params=tf.Variable(tf.zeros([T, 1, nitem])))

inference = ed.HMC({trait: q_trait, thresh: q_thresh}, data={X: X_data})
inference.run(step_size=0.1)

# Alternatively, use variational inference.
# q_trait = Normal(
#     loc=tf.Variable(tf.random_normal([nsubj, 1])),
#     scale=tf.nn.softplus(tf.Variable(tf.random_normal([nsubj, 1]))))
# q_thresh = Normal(
#     loc=tf.Variable(tf.random_normal([1, nitem])),
#     scale=tf.nn.softplus(tf.Variable(tf.random_normal([1, nitem]))))

# inference = ed.KLqp({trait: q_trait, thresh: q_thresh}, data={X: X_data})
# inference.run(n_iter=2500, n_samples=10)

# CRITICISM
# Check that the inferred posterior mean captures the true traits.
plt.scatter(trait_true, q_trait.mean().eval())
plt.show()

print("MSE between true traits and inferred posterior mean:")
print(np.mean(np.square(trait_true - q_trait.mean().eval())))
