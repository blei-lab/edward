"""Convolutional variational auto-encoder for binarized MNIST.

References
----------
http://edwardlib.org/tutorials/decoder
http://edwardlib.org/tutorials/inference-networks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import Progbar
from observations import mnist
from scipy.misc import imsave

tf.flags.DEFINE_string("data_dir", default="/tmp/data", help="")
tf.flags.DEFINE_string("out_dir", default="/tmp/out", help="")
tf.flags.DEFINE_integer("M", default=128, help="Batch size during training.")
tf.flags.DEFINE_integer("d", default=10, help="Latent dimension.")
tf.flags.DEFINE_integer("n_epoch", default=100, help="")

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
    batch = batch.astype(np.float32) / 255.0  # normalize pixel intensities
    batch = np.random.binomial(1, batch)  # binarize images
    yield batch


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  net = tf.reshape(z, [FLAGS.M, 1, 1, FLAGS.d])
  net = tf.layers.conv2d_transpose(net, 128, 3, padding='VALID')
  net = tf.layers.batch_normalization(net)
  net = tf.nn.elu(net)
  net = tf.layers.conv2d_transpose(net, 64, 5, padding='VALID')
  net = tf.layers.batch_normalization(net)
  net = tf.nn.elu(net)
  net = tf.layers.conv2d_transpose(net, 32, 5, strides=2, padding='SAME')
  net = tf.layers.batch_normalization(net)
  net = tf.nn.elu(net)
  net = tf.layers.conv2d_transpose(net, 1, 5, strides=2, padding='SAME')
  net = tf.reshape(net, [FLAGS.M, -1])
  return net


def inference_network(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  loc, scale = neural_network(x)
  """
  net = tf.reshape(x, [FLAGS.M, 28, 28, 1])
  net = tf.layers.conv2d(net, 32, 5, strides=2, padding='SAME')
  net = tf.layers.batch_normalization(net)
  net = tf.nn.elu(net)
  net = tf.layers.conv2d(net, 64, 5, strides=2, padding='SAME')
  net = tf.layers.batch_normalization(net)
  net = tf.nn.elu(net)
  net = tf.layers.conv2d(net, 128, 5, padding='VALID')
  net = tf.layers.batch_normalization(net)
  net = tf.nn.elu(net)
  net = tf.layers.dropout(net, 0.1)
  net = tf.reshape(net, [FLAGS.M, -1])
  net = tf.layers.dense(net, FLAGS.d * 2, activation=None)
  loc = net[:, :FLAGS.d]
  scale = tf.nn.softplus(net[:, FLAGS.d:])
  return loc, scale


def main(_):
  ed.set_seed(42)

  # DATA. MNIST batches are fed at training time.
  (x_train, _), (x_test, _) = mnist(FLAGS.data_dir)
  x_train_generator = generator(x_train, FLAGS.M)

  # MODEL
  z = Normal(loc=tf.zeros([FLAGS.M, FLAGS.d]),
             scale=tf.ones([FLAGS.M, FLAGS.d]))
  logits = generative_network(z)
  x = Bernoulli(logits=logits)

  # INFERENCE
  x_ph = tf.placeholder(tf.int32, [FLAGS.M, 28 * 28])
  loc, scale = inference_network(tf.cast(x_ph, tf.float32))
  qz = Normal(loc=loc, scale=scale)

  # Bind p(x, z) and q(z | x) to the same placeholder for x.
  inference = ed.KLqp({z: qz}, data={x: x_ph})
  optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
  inference.initialize(optimizer=optimizer)

  hidden_rep = tf.sigmoid(logits)

  tf.global_variables_initializer().run()

  n_iter_per_epoch = x_train.shape[0] // FLAGS.M
  for epoch in range(1, FLAGS.n_epoch + 1):
    print("Epoch: {0}".format(epoch))
    avg_loss = 0.0

    pbar = Progbar(n_iter_per_epoch)
    for t in range(1, n_iter_per_epoch + 1):
      pbar.update(t)
      x_batch = next(x_train_generator)
      info_dict = inference.update(feed_dict={x_ph: x_batch})
      avg_loss += info_dict['loss']

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss /= n_iter_per_epoch
    avg_loss /= FLAGS.M
    print("-log p(x) <= {:0.3f}".format(avg_loss))

    # Visualize hidden representations.
    images = hidden_rep.eval()
    for m in range(FLAGS.M):
      imsave(os.path.join(FLAGS.out_dir, '%d.png') % m,
             images[m].reshape(28, 28))

if __name__ == "__main__":
  tf.app.run()
