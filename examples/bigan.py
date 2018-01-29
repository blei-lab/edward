"""Adversarially Learned Inference (Dumoulin et al., 2017), aka
Bidirectional Generative Adversarial Networks (Donahue et al., 2017),
for joint learning of generator and inference networks for MNIST.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from observations import mnist

tf.flags.DEFINE_string("data_dir", default="/tmp/data", help="")
tf.flags.DEFINE_string("out_dir", default="/tmp/out", help="")
tf.flags.DEFINE_integer("M", default=100, help="Batch size during training.")
tf.flags.DEFINE_integer("d", default=50, help="Latent dimension.")
tf.flags.DEFINE_float("leak", default=0.2,
                      help="Leak parameter for leakyReLU.")
tf.flags.DEFINE_integer("hidden_units", default=300, help="")
tf.flags.DEFINE_float("encoder_variance", default=0.01,
                      help="Set to 0 for deterministic encoder.")

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


def leakyrelu(x, alpha=FLAGS.leak):
  return tf.maximum(x, alpha * x)


def gen_latent(x, hidden_units):
  net = tf.layers.dense(x, hidden_units, activation=leakyrelu)
  net = tf.layers.dense(net, FLAGS.d, activation=None)
  return (net + np.sqrt(FLAGS.encoder_variance) *
          np.random.normal(0.0, 1.0, [FLAGS.M, FLAGS.d]))


def gen_data(z, hidden_units):
  net = tf.layers.dense(z, hidden_units, activation=leakyrelu)
  net = tf.layers.dense(net, 784, activation=tf.sigmoid)
  return net


def discriminative_network(x, y):
  # Discriminator must output probability in logits
  net = tf.concat([x, y], 1)
  net = tf.layers.dense(net, FLAGS.hidden_units, activation=leakyrelu)
  net = tf.layers.dense(net, 1, activation=None)
  return net


def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  plt.title(str(samples))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig


def main(_):
  ed.set_seed(42)

  # DATA. MNIST batches are fed at training time.
  (x_train, _), (x_test, _) = mnist(FLAGS.data_dir)
  x_train_generator = generator(x_train, FLAGS.M)
  x_ph = tf.placeholder(tf.float32, [FLAGS.M, 784])
  z_ph = tf.placeholder(tf.float32, [FLAGS.M, FLAGS.d])

  # MODEL
  with tf.variable_scope("Gen"):
    xf = gen_data(z_ph, FLAGS.hidden_units)
    zf = gen_latent(x_ph, FLAGS.hidden_units)

  # INFERENCE:
  optimizer = tf.train.AdamOptimizer()
  optimizer_d = tf.train.AdamOptimizer()
  inference = ed.BiGANInference(
      latent_vars={zf: z_ph}, data={xf: x_ph},
      discriminator=discriminative_network)

  inference.initialize(
      optimizer=optimizer, optimizer_d=optimizer_d, n_iter=100000, n_print=3000)

  sess = ed.get_session()
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  idx = np.random.randint(FLAGS.M, size=16)
  i = 0
  for t in range(inference.n_iter):
    if t % inference.n_print == 1:

      samples = sess.run(xf, feed_dict={z_ph: z_batch})
      samples = samples[idx, ]
      fig = plot(samples)
      plt.savefig(os.path.join(FLAGS.out_dir, '{}{}.png').format(
          'Generated', str(i).zfill(3)), bbox_inches='tight')
      plt.close(fig)

      fig = plot(x_batch[idx, ])
      plt.savefig(os.path.join(FLAGS.out_dir, '{}{}.png').format(
          'Base', str(i).zfill(3)), bbox_inches='tight')
      plt.close(fig)

      zsam = sess.run(zf, feed_dict={x_ph: x_batch})
      reconstructions = sess.run(xf, feed_dict={z_ph: zsam})
      reconstructions = reconstructions[idx, ]
      fig = plot(reconstructions)
      plt.savefig(os.path.join(FLAGS.out_dir, '{}{}.png').format(
          'Reconstruct', str(i).zfill(3)), bbox_inches='tight')
      plt.close(fig)

      i += 1

    x_batch = next(x_train_generator)
    z_batch = np.random.normal(0, 1, [FLAGS.M, FLAGS.d])

    info_dict = inference.update(feed_dict={x_ph: x_batch, z_ph: z_batch})
    inference.print_progress(info_dict)

if __name__ == "__main__":
  tf.app.run()
