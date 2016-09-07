#!/usr/bin/env python
"""
Conditionally specified model.

As a proof of concept, we implement a conditionally specified model
with the structure

x | y ~ p(x | y)
y | x ~ p(y | x)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Normal


## MODEL SPECIFICATION


# We define a directed model, x_tied -> x -> y. The first and last nodes
# will be tied during inference.

x_tied = tf.Variable(0.0, trainable=False)
y = Normal(mu=x_tied, sigma=1.0)
x = Normal(mu=y, sigma=1.0)


## DATA GENERATION


init = tf.initialize_all_variables()
sess = ed.get_session()
sess.run(init)

assign_op = x_tied.assign(x.value())

# iteration 0 |
# x_tied's value is set to its initialization.
# Generate y^{(0)} ~ y | x_tied = 0.0.
# Generate x^{(0)} ~ x | y = y^{(0)}.
#
# iteration 1 |
# x_tied's value is assigned to x's value at iteration 0.
# Generate y^{(1)} ~ y | x_tied = x^{(0)}.
# Generate x^{(1)} ~ x | y = y^{(1)}.
#
# [...] induction

xy_samples = []
for t in range(100):
  x_sample, y_sample, _ = sess.run([x.value(), y.value(), assign_op])
  xy_samples += [(x_sample, y_sample)]

# importantly, note any function of x_tied uses the previous value of x.
# this preserves the right ordering for the fully conditional replications.
sess.run([x.value(), 0.0 + x_tied, assign_op])
## [-0.68678439, -2.1759129, -0.68678439]
sess.run([x.value(), 0.0 + x_tied, assign_op])
## [0.37869906, -0.68678439, 0.37869906]


## INFERENCE


# Here, we assume the model has no latent variables (and thus no
# priors/posteriors to infer). Inference will train parameters
# (tf.Variables) in the model.

data = {x: np.array(0.0), y: np.array(1.0)}
inference = ed.MAP([], data, tie={x_tied: x})

# Inference performs the following to calculate the loss:

# Copy random variables, swapping their dependencies according to
# dict_swap.
z_mode = {}
dict_swap = merge_dicts(z_mode, self.data)
y_copy = copy(y, dict_swap)  # x_tied -> copied/y
x_copy = copy(x, dict_swap)  # data[y] -> copied/x

# Calculate sum of fully conditional log-likelihoods.
p_log_prob = 0.0
p_log_prob += y_copy.log_prob(data[y])  # p(Y=data[y] | X=x_tied)
p_log_prob += x_copy.log_prob(data[x])  # p(X=data[x] | Y=data[y])


## CRITICISM


# Criticism remains the same. The user must work with the conditionals
# according to whatever they do.

# check against prediction of x, where prediction via x is the mean of
# p(x | y).
ed.evaluate('mean_squared_error', {x: np.array(0.0)})
