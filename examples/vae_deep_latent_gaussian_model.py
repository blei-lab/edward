#!/usr/bin/env python
"""
Probability model
    Deep latent Gaussian model
Variational model
    Likelihood: Mean-field Gaussian parameterized by inference network
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf

from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from deconv import deconv2d
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

class Variational:
    def encoder(self, input_tensor):
        '''Create encoder network.

        Args:
            input_tensor: a batch of flattened images [batch_size, 28*28]

        Returns:
            A tensor that expresses the encoder network
        '''
        return (pt.wrap(input_tensor).
                reshape([FLAGS.batch_size, 28, 28, 1]).
                conv2d(5, 32, stride=2).
                conv2d(5, 64, stride=2).
                conv2d(5, 128, edges='VALID').
                dropout(0.9).
                flatten().
                fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor

class Model:
    def decoder(self, input_tensor):
        '''Create decoder network.

        Args:
            input_tensor: a batch of vectors to decode

        Returns:
            A tensor that expresses the decoder network
        '''
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        mean = input_tensor[:, :FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:]))
        input_sample = mean + epsilon * stddev
        return (pt.wrap(input_sample).
                reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
                deconv2d(3, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
                flatten()).tensor, mean, stddev

    def decoder_none(self):
        '''Samples from a sampled vector.

        Returns:
            A tensor that expresses the decoder network
        '''
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        mean = None
        stddev = None
        input_sample = epsilon

        return (pt.wrap(input_sample).
                reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
                deconv2d(3, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
                flatten()).tensor, mean, stddev

class Inference:
    def get_vae_cost(self, mean, stddev, epsilon=1e-8):
        '''VAE loss
            See the paper

        Args:
            mean:
            stddev:
            epsilon:
        '''
        return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                    2.0 * tf.log(stddev + epsilon) - 1.0))

    def get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                             (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def build_loss(self):
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            with pt.defaults_scope(phase=pt.Phase.train):
                with tf.variable_scope("model") as scope:
                    output_tensor, mean, stddev = model.decoder(variational.encoder(input_tensor))

            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model", reuse=True) as scope:
                    sampled_tensor, _, _ = model.decoder_none()

        vae_loss = inference.get_vae_cost(mean, stddev)
        rec_loss = inference.get_reconstruction_cost(output_tensor, input_tensor)

        loss = vae_loss + rec_loss

variational = Variational()
model = Model()
inference = Inference()

input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])

loss = inference.build_loss()

## TRAIN

#data_directory = os.path.join(FLAGS.working_directory, "MNIST")
#if not os.path.exists(data_directory):
#    os.makedirs(data_directory)
#mnist = input_data.read_data_sets(data_directory, one_hot=True)

#optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
#train = pt.apply_optimizer(optimizer, losses=[loss])

#init = tf.initialize_all_variables()

#with tf.Session() as sess:
#    sess.run(init)

#    for epoch in range(FLAGS.max_epoch):
#        training_loss = 0.0

#        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
#        pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
#        pbar.start()
#        for i in range(FLAGS.updates_per_epoch):
#            pbar.update(i)
#            x, _ = mnist.train.next_batch(FLAGS.batch_size)
#            _, loss_value = sess.run([train, loss], {input_tensor: x})
#            training_loss += loss_value

#        training_loss = training_loss / \
#            (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

#        print("Loss %f" % training_loss)

#        imgs = sess.run(sampled_tensor)
#        for k in range(FLAGS.batch_size):
#            imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
#            if not os.path.exists(imgs_folder):
#                os.makedirs(imgs_folder)

#            imsave(os.path.join(imgs_folder, '%d.png') % k,
#                   imgs[k].reshape(28, 28))
