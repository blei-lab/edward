"""Sigmoid belief network (Neal, 1990) trained on the Caltech 101
Silhouettes data set.

Default settings take ~143s / epoch on a Titan X (Pascal). Results on
epoch 100:
Training negative log-likelihood: 209.443
Test negative log-likelihood: 161.244

Using n_train_samples=50 converges to test NLL of 157.824.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Bernoulli
from edward.util import Progbar
from observations import caltech101_silhouettes
from scipy.misc import imsave

tf.flags.DEFINE_string("data_dir", default="/tmp/data", help="")
tf.flags.DEFINE_string("out_dir", default="/tmp/out", help="")
tf.flags.DEFINE_integer("batch_size", default=24,
                        help="Batch size during training.")
tf.flags.DEFINE_list("hidden_sizes", default=[300, 100, 50, 10],
                     help="Hidden size per layer from bottom-up.")
tf.flags.DEFINE_integer("n_train_samples", default=10,
                        help="Number of samples for training.")
tf.flags.DEFINE_integer("n_test_samples", default=1000,
                        help="Number of samples to calculate test log-lik.")
tf.flags.DEFINE_float("step_size", default=1e-3,
                      help="Learning rate step size.")
tf.flags.DEFINE_integer("n_epoch", default=100, help="")
tf.flags.DEFINE_integer("n_iter_per_epoch", default=10000, help="")

FLAGS = tf.flags.FLAGS
if not os.path.exists(FLAGS.out_dir):
  os.makedirs(FLAGS.out_dir)


def generator(array, batch_size):
  """Generate batch with respect to array's first axis."""
  start = 0  # pointer to where we are in iteration
  while True:
    stop = start + batch_size
    diff = stop - array.shape[0]
    if diff <= 0:
      batch = array[start:stop]
      start += batch_size
    else:
      batch = np.concatenate((array[start:], array[:diff]))
      start = diff
    yield batch


def main(_):
  ed.set_seed(42)

  # DATA
  (x_train, _), (x_test, _), (x_valid, _) = caltech101_silhouettes(
      FLAGS.data_dir)
  x_train_generator = generator(x_train, FLAGS.batch_size)
  x_ph = tf.placeholder(tf.int32, [None, 28 * 28])

  # MODEL
  zs = [0] * len(FLAGS.hidden_sizes)
  for l in reversed(range(len(FLAGS.hidden_sizes))):
    if l == len(FLAGS.hidden_sizes) - 1:
      logits = tf.zeros([tf.shape(x_ph)[0], FLAGS.hidden_sizes[l]])
    else:
      logits = tf.layers.dense(tf.cast(zs[l + 1], tf.float32),
                               FLAGS.hidden_sizes[l], activation=None)
    zs[l] = Bernoulli(logits=logits)

  x = Bernoulli(logits=tf.layers.dense(tf.cast(zs[0], tf.float32),
                                       28 * 28, activation=None))

  # INFERENCE
  # Define variational model with reverse ordering as probability model:
  # if p is 15-100-300 from top-down, q is 300-100-15 from bottom-up.
  qzs = [0] * len(FLAGS.hidden_sizes)
  for l in range(len(FLAGS.hidden_sizes)):
    if l == 0:
      logits = tf.layers.dense(tf.cast(x_ph, tf.float32),
                               FLAGS.hidden_sizes[l], activation=None)
    else:
      logits = tf.layers.dense(tf.cast(qzs[l - 1], tf.float32),
                               FLAGS.hidden_sizes[l], activation=None)
    qzs[l] = Bernoulli(logits=logits)

  inference = ed.KLqp({z: qz for z, qz in zip(zs, qzs)}, data={x: x_ph})
  optimizer = tf.train.AdamOptimizer(FLAGS.step_size)
  inference.initialize(optimizer=optimizer, n_samples=FLAGS.n_train_samples)

  # Build tensor for log-likelihood given one variational sample to run
  # on test data.
  x_post = ed.copy(x, {z: qz for z, qz in zip(zs, qzs)})
  x_neg_log_prob = (-tf.reduce_sum(x_post.log_prob(x_ph)) /
                    tf.cast(tf.shape(x_ph)[0], tf.float32))

  sess = ed.get_session()
  tf.global_variables_initializer().run()

  for epoch in range(FLAGS.n_epoch):
    print("Epoch {}".format(epoch))
    train_loss = 0.0

    pbar = Progbar(FLAGS.n_iter_per_epoch)
    for t in range(1, FLAGS.n_iter_per_epoch + 1):
      pbar.update(t)
      x_batch = next(x_train_generator)
      info_dict = inference.update(feed_dict={x_ph: x_batch})
      train_loss += info_dict['loss']

    # Print per-data point loss, averaged over training epoch.
    train_loss /= FLAGS.n_iter_per_epoch
    train_loss /= FLAGS.batch_size
    print("Training negative log-likelihood: {:0.3f}".format(train_loss))

    test_loss = [sess.run(x_neg_log_prob, {x_ph: x_test})
                 for _ in range(FLAGS.n_test_samples)]
    test_loss = np.mean(test_loss)
    print("Test negative log-likelihood: {:0.3f}".format(test_loss))

    # Prior predictive check.
    images = sess.run(x, {x_ph: x_batch})  # feed ph to determine sample size
    for m in range(FLAGS.batch_size):
      imsave("{}/{}.png".format(out_dir, m), images[m].reshape(28, 28))

if __name__ == "__main__":
  tf.app.run()
