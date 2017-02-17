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


def discriminative_network(x, z):
  """Takes as input a minibatch of data (unused) and latent variable
  sample; outputs logit probabilities.
  """
  z = tf.reshape(z, [1, 1])  # reshape scalar as matrix input
  h1 = slim.fully_connected(z, 10, activation_fn=tf.nn.relu)
  logit = slim.fully_connected(h1, 1, activation_fn=None)
  return logit


ed.set_seed(42)

# MODEL
z = Normal(mu=5.0, sigma=1.0)

# INFERENCE
qz = Normal(mu=tf.Variable(tf.random_normal([])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

inference = ed.AVB({z: qz}, discriminator=discriminative_network)
inference.initialize(n_iter=1000)

sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  for _ in range(5):
    info_dict_d = inference.update(variables="Disc")

  info_dict = inference.update(variables="Gen")
  info_dict['loss_d'] = info_dict_d['loss_d']
  info_dict['t'] = info_dict['t'] // 6  # say set of 6 updates is 1 iteration

  t = info_dict['t']
  inference.print_progress(info_dict)
  if t % inference.n_print == 0:
    # Check inferred posterior parameters.
    mean, std = sess.run([qz.mean(), qz.std()])
    print("Inferred mean & std: {} {}".format(mean, std))
