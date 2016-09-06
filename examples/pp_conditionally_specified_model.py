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
from edward.util import copy


## MODEL SPECIFICATION


# We define a directed model, x_ph -> x -> y. The first and last nodes
# will be tied during inference.

x_ph = tf.constant(0.0)
y = Normal(mu=x_ph, sigma=1.0)
x = Normal(mu=y, sigma=1.0)


## DATA GENERATION


# There are two approaches.

# (1) Create the long chain of conditional samples. Then run the graph.
x_ph = tf.constant(0.0)
y = Normal(mu=x_ph, sigma=1.0)
xs = []
ys = []
for t in range(100):  # 100 samples
  x = Normal(mu=y, sigma=1.0)
  y = Normal(mu=x, sigma=1.0)
  xs += [x.value()]
  ys += [y.value()]

xy_samples = sess.run(xs + ys)

# (2) Create the undirected model. Then copy pieces of the model to
# make conditional samples. Then run the graph.
x_ph = tf.constant(0.0)
y = Normal(mu=x_ph, sigma=1.0)
x = Normal(mu=y, sigma=1.0)

y_sample = sess.run(y.value())
xy_samples = []
for t in range(100):  # 100 samples
  x_copy = copy(x, {y: y_sample}, deepcopy=True)
  y_copy = copy(y, {x_ph: x_copy}, deepcopy=True)
  x_sample, y_sample = sess.run([x_copy.value(), y_copy.value()])
  xy_samples += [(x_sample, y_sample)]


## INFERENCE


# Here, we assume the model has no latent variables (and thus no
# priors/posteriors to infer). Inference will train parameters
# (tf.Variables) in the model.

data = {x: np.array(0.0), y: np.array(1.0)}
inference = ed.MAP([], data, tie={x_ph: x})

# inference performs the following to calculate the loss:

y_copy = copy(y, {x_ph: x})
x_copy = copy(x, {x_ph: x})

p_log_prob = 0.0
p_log_prob += y_copy.log_prob(data[y])
p_log_prob += x_copy.log_prob(data[x])


## CRITICISM


# Criticism remains the same. The user must work with the conditionals
# according to whatever they do.

# check against prediction of x, where prediction via x is the mean of
# p(x | y).
ed.evaluate('mean_squared_error', {x: np.array(0.0)})
