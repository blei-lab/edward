#!/usr/bin/env python
"""Adversarial variational Bayes on a Gaussian posterior
(Mescheder et al., 2017).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal
from tensorflow.contrib import slim

# TODO what's random about this?
ed.set_seed(42)

def discriminative_network(x, z):
  """Outputs probability in logits.

  Takes as input a minibatch of data (unused) and latent variable
  sample.
  """
  # TODO
  # as a way to diagnose, maybe use the optimal discriminator here,
  # with a fake parameter
  z = tf.reshape(z, [1, 1])
  h1 = slim.fully_connected(z, 10, activation_fn=tf.nn.relu)
  logit = slim.fully_connected(h1, 1, activation_fn=None)
  return logit

# MODEL
z = Normal(mu=1.0, sigma=1.0)

# INFERENCE
qz = Normal(mu=tf.Variable(tf.random_normal([])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

inference = ed.AVB({z: qz}, discriminator=discriminative_network)
inference.initialize(n_iter=6000)

init = tf.global_variables_initializer()
init.run()

sess = ed.get_session()
for t in range(inference.n_iter // 6):
  for _ in range(5):
    inference.update(variables="Disc")

  info_dict = inference.update(variables="Gen")
  inference.print_progress(info_dict)
  if (t * 6) % inference.n_print == 0:
    # Check inferred posterior parameters.
    mean, std = sess.run([qz.mean(), qz.std()])
    print("Inferred mean & std: {} {}".format(mean, std))
