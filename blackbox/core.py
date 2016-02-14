from __future__ import print_function
import numpy as np
import tensorflow as tf

from scipy.stats import norm

class VI:
    """
    Base class for inference methods.

    Arguments
    ----------
    model: p(x, z), class with log_prob method
    q: q(z; lambda), class TODO
    n_iter: number of iterations for optimization
    n_minibatch: number of samples for stochastic gradient
    n_print: number of iterations for each print progress
    """
    def __init__(self, model, q,
                 n_iter=1000, n_minibatch=1, n_print=100):
        self.model = model
        self.q = q

        self.n_iter = n_iter
        self.n_minibatch = n_minibatch
        self.n_print = n_print

        self.samples = tf.placeholder(shape=(self.n_minibatch, q.num_vars),
                                      dtype=tf.float32,
                                      name="samples")
        self.elbo = 0

    def run(self):
        if self.q.reparam is False:
            loss = self.build_score_loss()
        else:
            loss = self.build_reparam_loss()

        update = tf.train.AdamOptimizer(0.1).minimize(-loss)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        for t in range(self.n_iter):
            if self.q.reparam is False:
                samples = self.q.sample(self.samples.get_shape(), sess)
            else:
                samples = self.q.sample_noise(self.samples.get_shape())

            _, elbos = sess.run([update, self.elbo], {self.samples: samples})

            self.print_progress(t, elbos, sess)

    def build_score_loss(self):
        # TODO use MFVI gradient
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, self.samples)

        self.elbo = self.model.log_prob(self.samples) - q_log_prob
        return tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbo))

    def build_reparam_loss(self):
        # TODO use MFVI gradient
        z = self.q.reparameterize(self.samples)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, z)

        self.elbo = self.model.log_prob(z) - q_log_prob
        # if isinstance(self.q, MFGaussian):
            # TODO calculate gaussian entropy analytically
        return tf.reduce_mean(self.elbo)

    def print_progress(self, t, elbos, sess):
        if t % self.n_print == 0:
            print("iter %d elbo %.2f " % (t, np.mean(elbos)))
            self.q.print_params(sess)

# TODO
# what portions of this should be part of the base class?
# how can I make MFVI a special case of this?
#class HVM(VI):
#    """
#    Black box inference with a hierarchical variational model.
#    (Ranganath et al., 2016)

#    Arguments
#    ----------
#    model: probability model p(x, z)
#    q_mf: likelihood q(z | lambda) (must be a mean-field)
#    q_prior: prior q(lambda; theta)
#    r_auxiliary: auxiliary r(lambda | z; phi)
#    """
#  def __init__(self, model, q_mf, q_prior, r_auxiliary,
#               *args, **kwargs):
#    VI.__init__(self, *args, **kwargs)
#    self.model = model
#    self.q_mf = q_mf
#    self.q_prior = q_prior
#    self.r_auxiliary = r_auxiliary

# TODO be consistent with tf.dtypes
# TODO visualize graph
