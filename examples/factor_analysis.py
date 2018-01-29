"""Logistic factor analysis on MNIST. Using Monte Carlo EM, with HMC
for the E-step and MAP for the M-step. We fit to just one data
point in MNIST.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import os
import tensorflow as tf

from edward.models import Bernoulli, Empirical, Normal
from observations import mnist
from scipy.misc import imsave

tf.flags.DEFINE_string("data_dir", default="/tmp/data", help="")
tf.flags.DEFINE_string("out_dir", default="/tmp/out", help="")
tf.flags.DEFINE_integer("N", default=1, help="Number of data points.")
tf.flags.DEFINE_integer("d", default=10, help="Number of latent dimensions.")
tf.flags.DEFINE_integer("n_iter_per_epoch", default=5000, help="")
tf.flags.DEFINE_integer("n_epoch", default=20, help="")

FLAGS = tf.flags.FLAGS
if not os.path.exists(FLAGS.out_dir):
  os.makedirs(FLAGS.out_dir)

FLAGS = tf.flags.FLAGS


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  net = tf.layers.dense(z, 28 * 28, activation=None)
  net = tf.reshape(net, [FLAGS.N, -1])
  return net


def main(_):
  ed.set_seed(42)

  # DATA
  (x_train, _), (x_test, _) = mnist(FLAGS.data_dir)
  x_train = x_train[:FLAGS.N]

  # MODEL
  z = Normal(loc=tf.zeros([FLAGS.N, FLAGS.d]),
             scale=tf.ones([FLAGS.N, FLAGS.d]))
  logits = generative_network(z)
  x = Bernoulli(logits=logits)

  # INFERENCE
  T = FLAGS.n_iter_per_epoch * FLAGS.n_epoch
  qz = Empirical(params=tf.get_variable("qz/params", [T, FLAGS.N, FLAGS.d]))

  inference_e = ed.HMC({z: qz}, data={x: x_train})
  inference_e.initialize()

  inference_m = ed.MAP(data={x: x_train, z: qz.params[inference_e.t]})
  optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
  inference_m.initialize(optimizer=optimizer)

  tf.global_variables_initializer().run()

  for _ in range(FLAGS.n_epoch - 1):
    avg_loss = 0.0
    for _ in range(FLAGS.n_iter_per_epoch):
      info_dict_e = inference_e.update()
      info_dict_m = inference_m.update()
      avg_loss += info_dict_m['loss']
      inference_e.print_progress(info_dict_e)

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss = avg_loss / FLAGS.n_iter_per_epoch
    avg_loss = avg_loss / FLAGS.N
    print("\nlog p(x) >= {:0.3f}".format(avg_loss))

    # Prior predictive check.
    images = x.eval()
    for m in range(FLAGS.N):
      imsave(os.path.join(FLAGS.out_dir, '%d.png') % m,
             images[m].reshape(28, 28))

if __name__ == "__main__":
  tf.app.run()
