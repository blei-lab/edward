#!/usr/bin/env python
"""
Probability model
    Prior: Normal
    Likelihood: Bernoulli parameterized by neural network
Variational model
    Likelihood: Mean-field Gaussian parameterized by neural network
"""
from __future__ import absolute_import, division, print_function
import os
import prettytensor as pt
import tensorflow as tf

from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from vae_util import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")

FLAGS = flags.FLAGS

def kl_gaussian(mean, stddev):
    """
    KL( N(z; mean, stddev) || N(z; 0, 1) )

    Parameters
    ----------
    assumes a matrix of each
    """
    return -0.5 * tf.reduce_sum(1.0 + 2.0 * tf.log(stddev + 1e-8) - \
                                tf.square(mean) - tf.square(stddev))

class Variational:
    def __init__(self):
        self.mean = None
        self.stddev = None

    def network(self, x):
        """
        mean, stddev | x = phi(x)
        where mean and stdddev are each batch_size x hidden_size
        """
        output = (pt.wrap(x).
                reshape([FLAGS.batch_size, 28, 28, 1]).
                conv2d(5, 32, stride=2).
                conv2d(5, 64, stride=2).
                conv2d(5, 128, edges='VALID').
                dropout(0.9).
                flatten().
                fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor
        self.mean = output[:, :FLAGS.hidden_size]
        self.stddev = tf.sqrt(tf.exp(output[:, FLAGS.hidden_size:]))

    def sample(self, x):
        """
        mean, stddev | x = phi(x)
        z | mean, stddev ~ N(0 | mean, stddev)

        Parameters
        ----------
        x : tf.Tensor
            a batch of flattened images [batch_size, 28*28]
        """
        self.network(x)
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        return self.mean + self.epsilon * stddev

class Model:
    def network(self, z):
        """
        p | z = varphi(z)
        """
        return (pt.wrap(z).
                reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
                deconv2d(3, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
                flatten()).tensor

    def sample_prior(self):
        """
        z ~ N(0, 1)
        """
        return tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])

    def log_likelihood(self, x, z):
        """
        p | z = varphi(z)
        log Bernoulli(x | p)
        """
        p = model.network(z)
        return x * tf.log(p + 1e-8) + (1.0 - x) * tf.log(1.0 - p + 1e-8)

class Inference:
    def __init__(self, model, variational):
        self.model = model
        self.variational = variational

        self.p_test = None
        self.loss = None
        self.train = None

    def init(self):
        self.loss = inference.build_loss()
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
        self.train = pt.apply_optimizer(optimizer, losses=[self.loss])

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        return sess

    def build_loss(self):
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            z = self.variational.sample(x)

            # Prior predictive check at test time
            z_test = self.model.sample_prior()
            self.p_test = self.model.network(z_test)

        # E_{q(z | x)} [ log p(x | z) ] - KL(q(z | x) || p(z))
        elbo = tf.reduce_sum(self.model.log_likelihood(x, z)) - \
               kl_gaussian(self.variational.mean, self.variational.stddev)
        return -elbo


variational = Variational()
model = Model()
inference = Inference(model, variational)

data_directory = os.path.join(FLAGS.working_directory, "data/mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
mnist = input_data.read_data_sets(data_directory, one_hot=True)

x = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])
sess = inference.init()
for epoch in range(FLAGS.max_epoch):
    training_loss = 0.0

    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
    pbar.start()
    for i in range(FLAGS.updates_per_epoch):
        pbar.update(i)
        x_batch, _ = mnist.train.next_batch(FLAGS.batch_size)
        _, loss_value = sess.run([inference.train, inference.loss], {x: x_batch})
        training_loss += loss_value

    training_loss = training_loss / \
        (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

    print("Loss %f" % training_loss)

    imgs = sess.run(inference.p_test)
    for k in range(FLAGS.batch_size):
        img_folder = os.path.join(FLAGS.working_directory, 'img')
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        imsave(os.path.join(img_folder, '%d.png') % k,
               imgs[k].reshape(28, 28))
