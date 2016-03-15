from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.data import Data
from blackbox.util import log_sum_exp

class Inference:
    """
    Base class for inference methods.

    Arguments
    ----------
    model: Model
        probability model p(x, z)
    data: Data, optional
        data x
    """
    def __init__(self, model, data=Data()):
        self.model = model
        self.data = data

class MonteCarlo(Inference):
    """
    Base class for Monte Carlo methods.

    Arguments
    ----------
    model: Model
        probability model p(x, z)
    data: Data, optional
        data x
    """
    def __init__(self, *args, **kwargs):
        Inference.__init__(self, *args, **kwargs)

class VariationalInference(Inference):
    """
    Base class for variational inference methods.

    Arguments
    ----------
    model: Model
        probability model p(x, z)
    variational: Variational
        variational model q(z; lambda)
    data: Data, optional
        data x
    """
    def __init__(self, model, variational, data=Data()):
        Inference.__init__(self, model, data)
        self.variational = variational

    def run(self, *args, **kwargs):
        """
        A simple wrapper to run the inference algorithm.
        """
        sess = self.initialize(*args, **kwargs)
        for t in range(self.n_iter):
            loss = self.update(sess)
            self.print_progress(t, loss, sess)

    def initialize(self, n_iter=1000, n_data=None, n_print=100):
        """
        Initialize inference algorithm.

        Arguments
        ----------
        n_iter: int, optional
            Number of iterations for optimization.
        n_data: int, optional
            Number of samples for data subsampling. Default is to use all
            the data.
        n_print: int, optional
            Number of iterations for each print progress.
        """
        self.n_iter = n_iter
        self.n_data = n_data
        self.n_print = n_print

        self.losses = tf.constant(0.0)

        loss = self.build_loss()
        # Use ADAM with a decaying scale factor
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                            global_step,
                                            100, 0.9, staircase=True)
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(
            loss, global_step=global_step)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        return sess

    def update(self, sess):
        _, loss = sess.run([self.train, self.losses])
        return loss

    def print_progress(self, t, losses, sess):
        if t % self.n_print == 0:
            print("iter %d loss %.2f " % (t, np.mean(losses)))
            self.variational.print_params(sess)

    def build_loss(self):
        raise NotImplementedError()

class MFVI(VariationalInference):
# TODO this isn't MFVI so much as VI where q is analytic
    """
    Mean-field variational inference
    (Ranganath et al., 2014; Kingma and Welling, 2014)
    """
    def __init__(self, *args, **kwargs):
        VariationalInference.__init__(self, *args, **kwargs)

    def initialize(self, n_minibatch=1, score=None, *args, **kwargs):
        """
        Parameters
        ----------
        n_minibatch: int, optional
            Number of samples from variational model for calculating
            stochastic gradients.
        score: bool, optional
            Whether to force inference to use the score function
            gradient estimator. Otherwise default is to use the
            reparameterization gradient if available.
        """
        if score is None and hasattr(self.variational, 'reparam'):
            self.score = False
        else:
            self.score = True

        self.n_minibatch = n_minibatch
        self.samples = tf.placeholder(shape=(self.n_minibatch, self.variational.num_vars),
                                      dtype=tf.float32,
                                      name='samples')
        return VariationalInference.initialize(self, *args, **kwargs)

    def update(self, sess):
        if self.score:
            samples = self.variational.sample(self.samples.get_shape(), sess)
        else:
            samples = self.variational.sample_noise(self.samples.get_shape())

        _, loss = sess.run([self.train, self.losses], {self.samples: samples})
        return loss

    def build_loss(self):
        if self.score:
            return self.build_score_loss()
        else:
            return self.build_reparam_loss()

    def build_score_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        """
        if hasattr(self.variational, 'entropy'):
            # ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
            # where entropy is analytic
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.variational.num_vars):
                q_log_prob += self.variational.log_prob_zi(i, self.samples)

            p_log_prob = self.model.log_prob(x, self.samples)
            q_entropy = self.variational.entropy()
            self.losses = p_log_prob + q_entropy
            return tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) + \
                   q_entropy
        else:
            # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.variational.num_vars):
                q_log_prob += self.variational.log_prob_zi(i, self.samples)

            x = self.data.sample(self.n_data)
            self.losses = self.model.log_prob(x, self.samples) - q_log_prob
            return -tf.reduce_mean(q_log_prob * tf.stop_gradient(self.losses))

    def build_reparam_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        """
        z = self.variational.reparam(self.samples)

        if hasattr(self.variational, 'entropy'):
            # ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
            # where entropy is analytic
            x = self.data.sample(self.n_data)
            self.losses = self.model.log_prob(x, z) + self.variational.entropy()
        else:
            # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.variational.num_vars):
                q_log_prob += self.variational.log_prob_zi(i, z)

            x = self.data.sample(self.n_data)
            self.losses = self.model.log_prob(x, z) - q_log_prob

        return -tf.reduce_mean(self.losses)

class KLpq(VariationalInference):
    """
    Kullback-Leibler(posterior, approximation) minimization
    using adaptive importance sampling.
    """
    def __init__(self, *args, **kwargs):
        VariationalInference.__init__(self, *args, **kwargs)

    def initialize(self, n_minibatch=1, *args, **kwargs):
        self.n_minibatch = n_minibatch
        self.samples = tf.placeholder(shape=(self.n_minibatch, self.variational.num_vars),
                                      dtype=tf.float32,
                                      name='samples')
        return VariationalInference.initialize(self, *args, **kwargs)

    def update(self, sess):
        if self.score:
            samples = self.variational.sample(self.samples.get_shape(), sess)
        else:
            samples = self.variational.sample_noise(self.samples.get_shape())

        _, loss = sess.run([self.train, self.losses], {self.samples: samples})
        return loss

    def build_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        """
        # loss = E_{q(z; lambda)} [ w_norm(z; lambda) *
        #                           ( log p(x, z) - log q(z; lambda) ) ]
        # where w_norm(z; lambda) = w(z; lambda) / sum_z( w(z; lambda) )
        # and w(z; lambda) = p(x, z) / q(z; lambda)
        #
        # gradient = - E_{q(z; lambda)} [ w_norm(z; lambda) *
        #                                 grad_{lambda} log q(z; lambda) ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_vars):
            q_log_prob += self.variational.log_prob_zi(i, self.samples)

        # 1/B sum_{b=1}^B grad_log_q * w_norm
        # = 1/B sum_{b=1}^B grad_log_q * exp{ log(w_norm) }
        x = self.data.sample(self.n_data)
        log_w = self.model.log_prob(x, self.samples) - q_log_prob

        # normalized log importance weights
        log_w_norm = log_w - log_sum_exp(log_w)
        w_norm = tf.exp(log_w_norm)

        self.losses = w_norm * log_w
        return -tf.reduce_mean(q_log_prob * tf.stop_gradient(w_norm))

class MAP(VariationalInference):
    """
    Maximum a posteriori
    """
    def __init__(self, model, variational, data=Data(), num_params=None):
        # TODO: chack if variational family is pointmass
        VariationalInference.__init__(self, model, variational, data)

    def build_loss(self):
        z = self.variational.get_params()
        x = self.data.sample(self.n_data)
        self.losses = self.model.log_prob(x, z)
        return -tf.reduce_mean(self.losses)
