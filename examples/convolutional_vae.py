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
from edward import Variational, Normal
from edward.util import kl_multivariate_normal
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_vars", 10, "Number of latent variables.")
flags.DEFINE_integer("n_iter_per_epoch", 1000, "Number of iterations per epoch.")
flags.DEFINE_integer("n_epoch", 100, "Maximum number of epochs.")
flags.DEFINE_integer("n_data", 128, "Mini-batch size for data subsampling.")
flags.DEFINE_string("data_directory", "data/mnist", "Directory to store data.")
flags.DEFINE_string("img_directory", "img", "Directory to store sampled images.")

FLAGS = flags.FLAGS

# TODO
# This is a temporary fix so that we can leverage TensorFlow sampling,
# which is faster than realizing TensorFlow parameters and then
# running SciPy sampling.
def sample_noise(self, size):
    """
    eps = sample_noise() ~ s(eps)
    s.t. z = reparam(eps; lambda) ~ q(z | lambda)
    """
    return tf.random_normal(size)

Normal.sample_noise = sample_noise

def sample_noise(self, size):
    eps_layers = [layer.sample_noise((size[0], layer.num_vars))
                  for layer in self.layers]
    return tf.concat(0, eps_layers)

Variational.sample_noise = sample_noise

from edward import VariationalInference
def initialize(self, *args, **kwargs):
    self.x = tf.placeholder(tf.float32, [FLAGS.n_data, 28 * 28])
    self.losses = tf.constant(0.0)

    loss = self.build_loss()
    optimizer = tf.train.AdamOptimizer(1e-2, epsilon=1.0)
    self.train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    return sess

def update(self, sess):
    x = self.data.sample(self.n_data)
    _, loss_value = sess.run([self.train, self.losses], {self.x: x})
    return loss_value

def build_loss(self):
    # ELBO = E_{q(z | x)} [ log p(x | z) ] - KL(q(z | x) || p(z))
    with tf.variable_scope("model") as scope:
        self.variational.set_params(self.variational.mapping(self.x))
        eps = self.variational.sample_noise([self.n_data, self.variational.num_vars])
        z = self.variational.reparam(eps)
        self.losses = self.model.log_lik(self.x, z) - \
                      kl_multivariate_normal(self.variational.layers[0].m,
                                             self.variational.layers[0].s

    return -tf.reduce_sum(self.losses)

ed.MFVI.initialize = initialize
ed.MFVI.update = update
ed.MFVI.build_loss = build_loss

class NormalBernoulli:
    def __init__(self, num_vars):
        self.num_vars = num_vars

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
        z ~ N(0, 1), p = phi(z)
        """
        z = tf.random_normal(size)
        return self.mapping(z)

def mapping(self, x):
    """
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

# TODO
class Data:
    def __init__(self, data):
        self.mnist = data

    def sample(self, size):
        x_batch, _ = mnist.train.next_batch(size)
        return x_batch

ed.set_seed(42)
model = NormalBernoulli(FLAGS.num_vars)
variational = Variational()
Normal.mapping = mapping
variational.add(Normal(FLAGS.num_vars))

if not os.path.exists(FLAGS.data_directory):
    os.makedirs(FLAGS.data_directory)
mnist = input_data.read_data_sets(FLAGS.data_directory, one_hot=True)
data = Data(mnist)

inference = ed.MFVI(model, variational, data)
sess = inference.initialize(n_data=FLAGS.n_data)
with tf.variable_scope("model", reuse=True) as scope:
    p_rep = model.sample_prior([FLAGS.n_data, FLAGS.num_vars])

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

    # Print a lower bound to the average marginal likelihood for a single pixel.
    print("log p(x) >= %f" % avg_loss)

    imgs = sess.run(p_rep)
    for b in range(FLAGS.n_data):
        if not os.path.exists(FLAGS.img_directory):
            os.makedirs(FLAGS.img_directory)

        imsave(os.path.join(FLAGS.img_directory, '%d.png') % b,
               imgs[b].reshape(28, 28))
