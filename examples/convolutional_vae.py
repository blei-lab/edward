#!/usr/bin/env python
"""
Convolutional variational auto-encoder for MNIST data. The model is
written in TensorFlow, with neural networks using Pretty Tensor.

Probability model
    Prior: Normal
    Likelihood: Bernoulli parameterized by convolutional NN
Variational model
    Likelihood: Mean-field Normal parameterized by convolutional NN
"""
from __future__ import print_function
import os
import edward as ed
import prettytensor as pt
import tensorflow as tf

from convolutional_vae_util import deconv2d
from edward.models import Variational, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_integer("n_data", 128, "Mini-batch size for data subsampling.")
tf.flags.DEFINE_string("data_directory", "data/mnist", "Directory to store data.")
tf.flags.DEFINE_string("img_directory", "img", "Directory to store sampled images.")
FLAGS = tf.flags.FLAGS

class NormalBernoulli:
    """
    Each binarized pixel in an image is modeled by a Bernoulli
    likelihood. The success probability for each pixel is the output
    of a neural network that takes samples from a normal prior as
    input.

    p(x, z) = Bernoulli(x | p = varphi(z)) Normal(z; 0, I)
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars # number of latent variables

    def mapping(self, z):
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

    def log_lik(self, x, z):
        """
        log p(x | z) = log Bernoulli(x | p = varphi(z))
        """
        p = self.mapping(z)
        return x * tf.log(p + 1e-8) + (1.0 - x) * tf.log(1.0 - p + 1e-8)

    def sample_prior(self, size):
        """
        p ~ some complex distribution induced by
        z ~ N(0, 1), p = varphi(z)
        """
        z = tf.random_normal([size, self.num_vars])
        # Note the output of this is not prior samples, but just the
        # success probability, i.e., the hidden representation learned
        # by the neural network.
        return self.mapping(z)

def mapping(self, x):
    """
    Inference network to parameterize variational family. It takes
    data x as input and outputs the variational parameters lambda.

    lambda = phi(x)
    """
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        params = (pt.wrap(x).
                reshape([FLAGS.n_data, 28, 28, 1]).
                conv2d(5, 32, stride=2).
                conv2d(5, 64, stride=2).
                conv2d(5, 128, edges='VALID').
                dropout(0.9).
                flatten().
                fully_connected(self.num_vars * 2, activation_fn=None)).tensor

    mean = params[:, :self.num_vars]
    stddev = tf.sqrt(tf.exp(params[:, self.num_vars:]))
    return [mean, stddev]

Normal.mapping = mapping

class Data:
    def __init__(self, data):
        self.mnist = data

    def sample(self, size):
        x_batch, _ = mnist.train.next_batch(size)
        return x_batch

ed.set_seed(42)
model = NormalBernoulli(num_vars=10)

variational = Variational()
variational.add(Normal(model.num_vars))

if not os.path.exists(FLAGS.data_directory):
    os.makedirs(FLAGS.data_directory)
mnist = input_data.read_data_sets(FLAGS.data_directory, one_hot=True)
data = Data(mnist)

inference = ed.VAE(model, variational, data)
sess = inference.initialize(n_data=FLAGS.n_data)
with tf.variable_scope("model", reuse=True) as scope:
    p_rep = model.sample_prior(FLAGS.n_data)

n_epoch = 100
n_iter_per_epoch = 1000
for epoch in range(n_epoch):
    avg_loss = 0.0

    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
    pbar.start()
    for t in range(n_iter_per_epoch):
        pbar.update(t)
        loss = inference.update(sess)
        avg_loss += loss

    # Take average over all ELBOs during the epoch, and over minibatch
    # of data points (images).
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / FLAGS.n_data

    # Print a lower bound to the average marginal likelihood for an
    # image.
    print("log p(x) >= {:0.3f}".format(avg_loss))

    imgs = sess.run(p_rep)
    for b in range(FLAGS.n_data):
        if not os.path.exists(FLAGS.img_directory):
            os.makedirs(FLAGS.img_directory)

        imsave(os.path.join(FLAGS.img_directory, '%d.png') % b,
               imgs[b].reshape(28, 28))
