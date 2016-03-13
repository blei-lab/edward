#!/usr/bin/env python
"""
Probability model
    Prior: Normal
    Likelihood: Bernoulli parameterized by convolutional NN
Variational model
    Likelihood: Convolutional variational auto-encoder
                (Mean-field Gaussian parameterized by convolutional NN)
"""
from __future__ import division, print_function
import os
import prettytensor as pt
import tensorflow as tf
import blackbox as bb

from blackbox.util import kl_multivariate_normal
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from convolutional_vae_util import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")

FLAGS = flags.FLAGS

bb.set_seed(42)

class MFGaussian:
    def __init__(self):
        self.mean = None # batch_size x hidden_size
        self.stddev = None # batch_size x hidden_size

    def network(self, x):
        """
        mean, stddev = phi(x)
        """
        output = (pt.wrap(x).
                reshape([FLAGS.batch_size, 28, 28, 1]).
                conv2d(5, 32, stride=2).
                conv2d(5, 64, stride=2).
                conv2d(5, 128, edges='VALID').
                dropout(0.9).
                flatten().
                fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor
        return output

    def extract_params(self, output):
        #self.mean = output[:, :FLAGS.hidden_size]
        #self.stddev = tf.sqrt(tf.exp(output[:, FLAGS.hidden_size:]))
        mean = output[:, :FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(output[:, FLAGS.hidden_size:]))
        return mean, stddev

    def sample(self, x):
        """
        z | x ~ q(z | x) = N(z | mean, stddev = phi(x))

        Parameters
        ----------
        x : tf.Tensor
            a batch of flattened images [batch_size, 28*28]
        """
        extract_params(self.network(x))
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        return self.mean + epsilon * self.stddev

    def sample_ms(self, x):
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
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            z_test = self.sample_prior()
            return self.network(z_test)

class Data:
    def __init__(self, data):
        self.mnist = data

    def sample(self, size=FLAGS.batch_size):
        x_batch, _ = mnist.train.next_batch(size)
        return x_batch

class Inference:
    def __init__(self, model, variational, data):
        self.model = model
        self.variational = variational
        self.data = data

    def init(self):
        self.x = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])

        self.loss = inference.build_loss()
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
        self.train = pt.apply_optimizer(optimizer, losses=[self.loss])

        init = tf.initialize_all_variables()
        sess = tf.Session()
        # TODO there shouldn't be any of that variable creation stuff.
        # why was it hidden away before?
        sess.run(init)
        return sess

    def update(self, sess):
        x = self.data.sample()
        _, loss_value = sess.run([self.train, self.loss], {self.x: x})
        return loss_value

    def build_loss(self):
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            z = self.variational.sample(self.x)

            # TODO move this over to model
            z_test = self.model.sample_prior()
            self.p_test = self.model.network(z_test)

        # ELBO = E_{q(z | x)} [ log p(x | z) ] - KL(q(z | x) || p(z))
        # In general, there should be a scale factor due to data
        # subsampling, so that
        # ELBO = N / M * ( ELBO using x_b )
        # where x^b is a mini-batch of x, with sizes M and N respectively.
        # This is absorbed into the learning rate.
        elbo = tf.reduce_sum(self.model.log_likelihood(self.x, z)) - \
               kl_multivariate_normal(self.variational.mean, self.variational.stddev)
        return -elbo

    def init_temp(self, model, variational):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])

        loss, sampled_tensor = self.build_loss_temp(model, variational, x)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
        train = pt.apply_optimizer(optimizer, losses=[loss])

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        return sess, train, loss, x, sampled_tensor

    def update_temp(self, sess, train, loss, x, data):
        x_b = data.sample()
        _, loss_value = sess.run([train, loss], {x: x_b})
        return loss_value

    def build_loss_temp(self, model, variational, x):
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
            z, mean, stddev = variational.sample_ms(x)
            elbo = tf.reduce_sum(model.log_likelihood(x, z)) - \
                   kl_multivariate_normal(mean, stddev)
            #
            sampled_tensor = model.network(model.sample_prior())

        return -elbo, sampled_tensor

variational = MFGaussian()
model = NormalBernoulli()

data_directory = os.path.join(FLAGS.working_directory, "data/mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
mnist = input_data.read_data_sets(data_directory, one_hot=True)
data = Data(mnist)

inference = Inference(model, variational, data)

#sess = inference.init()
#sampled_tensor = inference.p_test
sess, train, loss, x, sampled_tensor = inference.init_temp(model, variational)

for epoch in range(FLAGS.max_epoch):
    avg_loss = 0.0

    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
    pbar.start()
    for i in range(FLAGS.updates_per_epoch):
        pbar.update(i)
        loss_value = inference.update_temp(sess, train, loss, x, data)
#        loss_value = inference.update(sess)
        avg_loss += loss_value

    avg_loss = avg_loss / \
        (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

    print("-log p(x) <= %f" % avg_loss)

    imgs = sess.run(sampled_tensor)
    for k in range(FLAGS.batch_size):
        imgs_folder = os.path.join(FLAGS.working_directory, 'img')
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        imsave(os.path.join(imgs_folder, '%d.png') % k,
               imgs[k].reshape(28, 28))
