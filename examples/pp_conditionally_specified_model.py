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




## INFERENCE


# Here, we assume the model has no latent variables (and thus no
# priors/posteriors to infer). Inference will train parameters
# (tf.Variables) in the model.

data = {x: np.array(0.0), y: np.array(1.0)}
inference = ed.MAP([], data, tie={x_tied: x})

# Inference performs the following to calculate the loss:

dict_swap = data
y_copy = copy(y, dict_swap)  # x_tied -> copied/y
x_copy = copy(x, dict_swap)  # data[y] -> copied/x

# (2) Perform calculations of the loss under thse conditionals.

# Calculate sum of fully conditional log-likelihoods.
p_log_prob = 0.0
p_log_prob += y_copy.log_prob(data[y])  # p(Y=data[y] | X=data[x])
p_log_prob += x_copy.log_prob(data[x])  # p(X=data[x] | Y=data[y])


## CRITICISM


# Criticism remains the same. The user must work with the conditionals
# according to whatever they do.

# check against prediction of x, where prediction via x is the mean of
# p(x | y).
ed.evaluate('mean_squared_error', {x: np.array(0.0)})


## ARCHIVES

# In general, just like with the implementation of directed models,
# there are two options for implementation of undirected models:
# variable assignment and graph copying.

# In this particular application, where we simply want to assign the
# tied node to whatever value its corresponding node will be, the
# grpah structure is fixed. This means that whereas graph copying is
# more general as it can change the graph structure, variable
# assignment will work here too.
#
# We appeal to graph copying, if only because we already do it for
# directed models and dealing with the same issues that arise there.

# Arguably in this case, variable assignment is
# more useful as the internal infrastructure because this drastically
# simplifies data generation (contrast above with previous approach).
# This is the only reason.
#
# note data generation is only typical of conditionally specified
# models, which are a special case of undirected models where the
# potentials in fact represent conditional distributions. (is that
# even true?)

# Under graph copying:
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


# Under variable assignment, there is one approach.

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

# During inference

# (1) Form new conditional random variables. Copy random variables,
# swapping their dependencies according to dict_swap.

# Swap all conditioning on prior to conditioning on posterior sample.
dict_swap = self.latent_vars
# Swap all conditioning on observed variable to conditioning on data.
for x, obs in six.iteritems(self.data):
  if isinstance(x, RandomVariable):
    dict_swap[x] = obs
# Swap all conditioning on a tied variable to conditioning on its
# corresponding posterior sample or data.
for rv_tie, rv in six.iteritems(self.tie):
  if rv in self.latent_vars:
    dict_swap[rv_tie] = self.latent_vars[rv]
  elif rv in self.data:
    dict_swap[rv_tie] = self.data[rv]
  else:
    raise IndexError("rv that it is tied to is not in either dict!")


y_copy = copy(y, {x_tied: data[tie[x_tied]]}) # during copy
x_tied.assign(data[tie[x_tied]]) # during initialization
