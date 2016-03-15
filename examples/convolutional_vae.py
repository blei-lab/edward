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

flags.DEFINE_integer("num_vars", 10, "Number of latent variables.")
flags.DEFINE_integer("n_iter_per_epoch", 1000, "Number of iterations per epoch.")
flags.DEFINE_integer("n_epoch", 100, "Maximum number of epochs.")
flags.DEFINE_integer("n_data", 128, "Mini-batch size for data subsampling.")
flags.DEFINE_string("data_directory", "data/mnist", "Directory to store data.")
flags.DEFINE_string("img_directory", "img", "Directory to store sampled images.")

FLAGS = flags.FLAGS

class MFGaussian:
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.mean = None # n_data x num_vars
        self.stddev = None # n_data x num_vars

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
                    reshape([FLAGS.n_data, 28, 28, 1]).
                    conv2d(5, 32, stride=2).
                    conv2d(5, 64, stride=2).
                    conv2d(5, 128, edges='VALID').
                    dropout(0.9).
                    flatten().
                    fully_connected(self.num_vars * 2, activation_fn=None)).tensor

    def extract_params(self, params):
        self.mean = params[:, :self.num_vars]
        self.stddev = tf.sqrt(tf.exp(params[:, self.num_vars:]))

    def sample(self, size, x):
        """
        z | x ~ q(z | x) = N(z | mean, stddev = phi(x))

        Parameters
        ----------
        x : tf.Tensor
            a batch of flattened images [n_data, 28*28]
        """
        self.extract_params(self.network(x))
        epsilon = tf.random_normal(size)
        return self.mean + epsilon * self.stddev

class NormalBernoulli:
    def __init__(self, num_vars):
        self.num_vars = num_vars

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
                    reshape([FLAGS.n_data, 1, 1, self.num_vars]).
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

    def sample_prior(self, size):
        """
        z ~ N(0, 1)
        """
        return tf.random_normal(size)

    def sample_latent(self, size):
        # Prior predictive check at test time
        z_rep = self.sample_prior(size)
        return self.network(z_rep)

class Data:
    def __init__(self, data):
        self.mnist = data

    def sample(self, size):
        x_batch, _ = mnist.train.next_batch(size)
        return x_batch

bb.set_seed(42)
model = NormalBernoulli(FLAGS.num_vars)
variational = MFGaussian(FLAGS.num_vars)

if not os.path.exists(FLAGS.data_directory):
    os.makedirs(FLAGS.data_directory)
mnist = input_data.read_data_sets(FLAGS.data_directory, one_hot=True)
data = Data(mnist)

inference = bb.VAE(model, variational, data)
sess = inference.init(n_data=FLAGS.n_data)
with tf.variable_scope("model", reuse=True) as scope:
    p_rep = model.sample_latent([FLAGS.n_data, FLAGS.num_vars])

for epoch in range(FLAGS.n_epoch):
    avg_loss = 0.0

    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.n_iter_per_epoch, widgets=widgets)
    pbar.start()
    for t in range(FLAGS.n_iter_per_epoch):
        pbar.update(t)
        loss = inference.update(sess)
        avg_loss += loss

    # Take average of all ELBOs during the epoch.
    avg_loss = avg_loss / FLAGS.n_iter_per_epoch
    # Take average over each data point (pixel), where each image has
    # 28*28 pixels.
    avg_loss = avg_loss / (28 * 28 * FLAGS.n_data)

    # Print (an upper bound to) the average NLL for a single pixel.
    print("-log p(x) <= %f" % avg_loss)

    imgs = sess.run(p_rep)
    for b in range(FLAGS.n_data):
        if not os.path.exists(FLAGS.img_directory):
            os.makedirs(FLAGS.img_directory)

        imsave(os.path.join(FLAGS.img_directory, '%d.png') % b,
               imgs[b].reshape(28, 28))
