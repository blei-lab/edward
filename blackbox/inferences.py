from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.data import Data

class Inference:
    """
    Base class for inference methods.

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
        self.model = model
        self.variational = variational
        self.data = data

    def run(self, n_iter=1000, n_minibatch=1, n_data=None, n_print=100):
        """
        Run inference algorithm.

        Arguments
        ----------
        n_iter: int, optional
            Number of iterations for optimization.
        n_minibatch: int, optional
            Number of samples from variational model for calculating
            stochastic gradients.
        n_data: int, optional
            Number of samples for data subsampling. Default is to use all
            the data.
        n_print: int, optional
            Number of iterations for each print progress.
        """
        self.n_iter = n_iter
        self.n_minibatch = n_minibatch
        self.n_data = n_data
        self.n_print = n_print
        self.samples = tf.placeholder(shape=(self.n_minibatch, self.variational.num_vars),
                                      dtype=tf.float32,
                                      name='samples')
        self.elbos = tf.zeros([self.n_minibatch])

        if hasattr(self.variational, 'reparam'):
            loss = self.build_reparam_loss()
        else:
            loss = self.build_score_loss()

        update = tf.train.AdamOptimizer(0.1).minimize(loss)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        for t in range(self.n_iter):
            if hasattr(self.variational, 'reparam'):
                samples = self.variational.sample_noise(self.samples.get_shape())
            else:
                samples = self.variational.sample(self.samples.get_shape(), sess)

            _, elbo = sess.run([update, self.elbos], {self.samples: samples})

            self.print_progress(t, elbo, sess)

    def print_progress(self, t, elbos, sess):
        if t % self.n_print == 0:
            print("iter %d elbo %.2f " % (t, np.mean(elbos)))
            self.variational.print_params(sess)

    def build_score_loss(self):
        pass

    def build_reparam_loss(self):
        pass

class MFVI(Inference):
# TODO this isn't MFVI so much as VI where q is analytic
    """
    Mean-field variational inference
    (Ranganath et al., 2014; Kingma and Welling, 2014)
    """
    def __init__(self, *args, **kwargs):
        VI.__init__(self, *args, **kwargs)

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

            x = self.data.sample(self.n_data)
            p_log_prob = self.model.log_prob(x, self.samples)
            q_entropy = self.variational.entropy()
            self.elbos = p_log_prob + q_entropy
            return tf.reduce_mean(q_log_prob * \
                                  tf.stop_gradient(p_log_prob)) + \
                   q_entropy
        else:
            # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.variational.num_vars):
                q_log_prob += self.variational.log_prob_zi(i, self.samples)

            x = self.data.sample(self.n_data)
            self.elbos = self.model.log_prob(x, self.samples) - \
                         q_log_prob
            return -tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbos))

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
            self.elbos = tf.reduce_mean(
                self.model.log_prob(x, z)) + self.variational.entropy()
        else:
            # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.variational.num_vars):
                q_log_prob += self.variational.log_prob_zi(i, z)

            x = self.data.sample(self.n_data)
            self.elbos = tf.reduce_mean(
                self.model.log_prob(x, z) - q_log_prob)

        return -self.elbos

class AlphaVI(Inference):
    """
    alpha-divergence.
    (Dustin's version, not Li et al. (2016)'s)

    Arguments
    ----------
    alpha: scalar in [0,1) U (1, infty)
    """
    def __init__(self, alpha, *args, **kwargs):
        VI.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def build_score_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        """
        # ELBO = E_{q(z; lambda)} [ w(z; lambda)^{1-alpha} ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_vars):
            q_log_prob += self.variational.log_prob_zi(i, self.samples)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        x = self.data.sample(self.n_data)
        log_omega = self.model.log_prob(x, self.samples) - \
                    q_log_prob
        max_log_omega = tf.reduce_max(log_omega)
        self.elbos = tf.pow(
            tf.exp(max_log_omega)*tf.exp(log_omega - max_log_omega),
            1.0-self.alpha)
        loss = tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbos))
        if self.alpha < 1:
            return -loss
        else:
            return loss

    def build_reparam_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        """
        z = self.variational.reparam(self.samples)
        # ELBO = E_{q(z; lambda)} [ w(z; lambda)^{1-alpha} ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_vars):
            q_log_prob += self.variational.log_prob_zi(i, z)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        x = self.data.sample(self.n_data)
        log_omega = self.model.log_prob(x, z) - \
                    q_log_prob
        max_log_omega = tf.reduce_max(log_omega)
        self.elbos = tf.pow(
            tf.exp(max_log_omega)*tf.exp(log_omega - max_log_omega),
            1.0-self.alpha)
        loss = (1.0-self.alpha) * \
               tf.reduce_mean(log_omega * tf.stop_gradient(self.elbos))
        if self.alpha < 1:
            return -loss
        else:
            return loss

    def print_progress(self, t, elbos, sess):
        if t % self.n_print == 0:
            elbo = np.mean(elbos)
            lower_bound = 1.0 / (self.alpha - 1.0) * np.log(elbo)
            print("iter %d elbo %.2f " % (t, lower_bound))
            self.variational.print_params(sess)

class LiVI(Inference):
    """
    alpha-divergence.
    A generalized version of Li et al. (2016)'s gradient, enabling
    score function gradient and minibatches (but no importance
    samples).
    The gradients turn out to be the same as Dustin's approach up to
    constants, although the score function gradient is slightly
    different.

    Arguments
    ----------
    alpha: scalar in [0,1) U (1, infty)
    """
    def __init__(self, alpha, *args, **kwargs):
        VI.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def build_score_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        """
        # ELBO = E_{q(z; lambda)} [ log w(z; lambda)^{1-alpha} ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_vars):
            q_log_prob += self.variational.log_prob_zi(i, self.samples)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        x = self.data.sample(self.n_data)
        log_omega = self.model.log_prob(x, self.samples) - \
                    q_log_prob
        max_log_omega = tf.reduce_max(log_omega)
        self.elbos = tf.pow(
            tf.exp(max_log_omega)*tf.exp(log_omega - max_log_omega),
            1.0-self.alpha)
        loss = tf.reduce_mean(q_log_prob *
                   tf.stop_gradient(tf.pow(log_omega, 1.0-self.alpha)))
        if self.alpha < 1:
            return -loss
        else:
            return loss

    def build_reparam_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        """
        z = self.variational.reparam(self.samples)
        # ELBO = E_{q(z; lambda)} [ log w(z; lambda)^{1-alpha} ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_vars):
            q_log_prob += self.variational.log_prob_zi(i, z)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        x = self.data.sample(self.n_data)
        log_omega = self.model.log_prob(x, self.samples) - \
                    q_log_prob
        max_log_omega = tf.reduce_max(log_omega)
        self.elbos = tf.pow(
            tf.exp(max_log_omega)*tf.exp(log_omega - max_log_omega),
            1.0-self.alpha)
        loss = (1.0-self.alpha) * \
               tf.reduce_mean(log_omega * tf.stop_gradient(self.elbos))
        if self.alpha < 1:
            return -loss
        else:
            return loss

    def print_progress(self, t, elbos, sess):
        if t % self.n_print == 0:
            elbo = np.mean(elbos)
            lower_bound = 1.0 / (self.alpha - 1.0) * np.log(elbo)
            print("iter %d elbo %.2f " % (t, lower_bound))
            self.variational.print_params(sess)

# TODO
# what portions of this should be part of the base class?
# how can I make MFVI a special case of this?
#class HVM(Inference):
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
