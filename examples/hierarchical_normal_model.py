#!/usr/bin/env python
"""Coaching effects from parallel experiments conducted in 8 schools
(Rubin, 1981; Section 5.5 in Gelman et al., 2003).

Assuming a Stan program "8schools.stan", the equivalent command is
Stan (RStan) is

library(rstan)
schools_dat <- list(J = 8,
             K = 2,
             y = c(28,  8, -3,  7, -1,  1, 18, 12),
             sigma = c(15, 10, 16, 11,  9, 11, 10, 18))
fit <- stan(file="8schools.stan", data=schools_dat,
            iter=10000, chains=1, algorithm="HMC", control=list(stepsize=0.2))
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Normal, Uniform

ed.set_seed(42)

# DATA
J = 8  # number of schools
y_data = np.array([28,  8, -3,  7, -1,  1, 18, 12]).astype(np.float32)
sigma_data = np.array([15, 10, 16, 11,  9, 11, 10, 18]).astype(np.float32)

# MODEL
# TODO is this the way to handle flat?
# + we could probably have a real flat prior, and not implement sample
# for it; we would say that the model can't be sampled from but can
# still be stacked with other random variables; methods which work on
# them cannot sample
mu = Uniform(a=-100.0, b=100.0)
tau = Uniform(a=-100.0, b=100.0)
theta = Normal(mu=mu * tf.ones(J), sigma=tau * tf.ones(J))
y = Normal(mu=theta, sigma=sigma_data)

# INFERENCE
T = 7500  # number of empirical samples
# qmu = Empirical(params=tf.Variable(tf.random_normal([T])))
# qtau = Empirical(params=tf.Variable(tf.random_normal([T])))
# qtheta = Empirical(params=tf.Variable(tf.random_normal([T, J])))
qmu = Empirical(params=tf.Variable(7.9 * tf.ones([T])))
qtau = Empirical(params=tf.Variable(6.3 * tf.ones([T])))
qtheta = Empirical(params=tf.Variable(5.0 * tf.ones([T, J])))

inference = ed.HMC({mu: qmu, tau: qtau, theta: qtheta}, data={y: y_data})
inference.run(step_size=0.75)

# CRITICISM
sess = ed.get_session()
# TODO to warm-up or not to warm-up
# qmu_mean, qtau_mean, qtheta_mean = sess.run(
#     [qmu.mean(), qtau.mean(), qtheta.mean()])
qmu_mean, qtau_mean, qtheta_mean = sess.run(
    [tf.reduce_mean(qmu.params[500:]),
     tf.reduce_mean(qtau.params[500:]),
     tf.reduce_mean(qtheta.params[500:], 0)])

print("Estimated shared effect:")
print(qmu_mean)
print("Estimated shared deviation:")
print(qtau_mean)
print("Estimated coaching effects in each school:")
print(qtheta_mean)
