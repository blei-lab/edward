#!/usr/bin/env python
"""
Convolutional variational auto-encoder for MNIST data. The model is
written in TensorFlow, with neural networks using Pretty Tensor.

Probability model
    Prior: Normal
    Likelihood: Bernoulli parameterized by convolutional NN
Variational model
    Likelihood: Mean-field Gaussian parameterized by convolutional NN
"""
from __future__ import division, print_function
import os
import prettytensor as pt
import tensorflow as tf
import blackbox as bb

from convolutional_vae_util import deconv2d
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")

FLAGS = flags.FLAGS

class MFGaussian:
    def __init__(self):
        self.mean = None # batch_size x hidden_size
        self.stddev = None # batch_size x hidden_size

    def network(self, x):
        """
        mean, stddev = phi(x)
        """
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            return (pt.wrap(x).
                    reshape([FLAGS.batch_size, 28, 28, 1]).
                    conv2d(5, 32, stride=2).
                    conv2d(5, 64, stride=2).
                    conv2d(5, 128, edges='VALID').
                    dropout(0.9).
                    flatten().
                    fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor

    # TODO in general, think about global vs local stuff
    #def extract_params(self, output):
    #    self.mean = output[:, :FLAGS.hidden_size]
    #    self.stddev = tf.sqrt(tf.exp(output[:, FLAGS.hidden_size:]))

    #def sample(self, x):
    #    """
    #    z | x ~ q(z | x) = N(z | mean, stddev = phi(x))

    #    Parameters
    #    ----------
    #    x : tf.Tensor
    #        a batch of flattened images [batch_size, 28*28]
    #    """
    #    self.extract_params(self.network(x))
    #    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    #    return self.mean + epsilon * self.stddev

    def extract_params(self, output):
        mean = output[:, :FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(output[:, FLAGS.hidden_size:]))
        return mean, stddev

    def sample(self, x):
        output = self.network(x)
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        mean, stddev = self.extract_params(output)
        z = mean + epsilon * stddev
        return z, mean, stddev

class NormalBernoulli:
    def network(self, z):
        """
        p = varphi(z)
        """
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            return (pt.wrap(z).
                    reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
                    deconv2d(3, 128, edges='VALID').
                    deconv2d(5, 64, edges='VALID').
                    deconv2d(5, 32, stride=2).
                    deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
                    flatten()).tensor

    def log_likelihood(self, x, z):
        """
        log p(x | z) = log Bernoulli(x | p = varphi(z))
        """
        p = self.network(z)
        return x * tf.log(p + 1e-8) + (1.0 - x) * tf.log(1.0 - p + 1e-8)

    def sample_prior(self):
        """
        z ~ N(0, 1)
        """
        return tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])

    def sample_latent(self):
        # Prior predictive check at test time
        z_rep = self.sample_prior()
        return self.network(z_rep)

class Data:
    def __init__(self, data):
        self.mnist = data

    def sample(self, size=FLAGS.batch_size):
        x_batch, _ = mnist.train.next_batch(size)
        return x_batch

bb.set_seed(42)
variational = MFGaussian()
model = NormalBernoulli()

data_directory = os.path.join(FLAGS.working_directory, "data/mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
mnist = input_data.read_data_sets(data_directory, one_hot=True)
data = Data(mnist)

inference = bb.VAE(model, variational, data)
sess = inference.init(n_data=FLAGS.batch_size)
with tf.variable_scope("model", reuse=True) as scope:
    p_rep = model.sample_latent()

for epoch in range(FLAGS.max_epoch):
    avg_loss = 0.0

    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
    pbar.start()
    for t in range(FLAGS.updates_per_epoch):
        pbar.update(t)
        loss = inference.update(sess)
        avg_loss += loss

    # Take average of all ELBOs during the epoch.
    avg_loss = avg_loss / FLAGS.updates_per_epoch
    # Take average over each data point (pixel), where each image has
    # 28*28 pixels.
    avg_loss = avg_loss / (28 * 28 * FLAGS.batch_size)

    # Print (an upper bound to) the average NLL for a single pixel.
    print("-log p(x) <= %f" % avg_loss)

    imgs = sess.run(p_rep)
    for b in range(FLAGS.batch_size):
        img_folder = os.path.join(FLAGS.working_directory, 'img')
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        imsave(os.path.join(img_folder, '%d.png') % b,
               imgs[b].reshape(28, 28))
