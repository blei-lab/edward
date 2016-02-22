from __future__ import print_function
import numpy as np
import tensorflow as tf

class VI:
    """
    Base class for inference methods.

    Arguments
    ----------
    model: p(x, z), class with log_prob method
    q: q(z; lambda), class TODO
    """
    def __init__(self, model, q):
        self.model = model
        self.q = q

    def run(self, n_iter=1000, n_minibatch=1, n_print=100):
        """
        Run inference algorithm.

        Arguments
        ----------
        n_iter: number of iterations for optimization
        n_minibatch: number of samples for stochastic gradient
        n_print: number of iterations for each print progress
        """
        self.n_iter = n_iter
        self.n_minibatch = n_minibatch
        self.n_print = n_print
        self.samples = tf.placeholder(shape=(self.n_minibatch, self.q.num_vars),
                                      dtype=tf.float32,
                                      name='samples')
        self.elbos = tf.zeros([self.n_minibatch])

        if hasattr(self.q, 'reparam'):
            loss = self.build_reparam_loss()
        else:
            loss = self.build_score_loss()

        update = tf.train.AdamOptimizer(0.1).minimize(loss)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        for t in range(self.n_iter):
            if hasattr(self.q, 'reparam'):
                samples = self.q.sample_noise(self.samples.get_shape())
            else:
                samples = self.q.sample(self.samples.get_shape(), sess)

            _, elbo = sess.run([update, self.elbos], {self.samples: samples})

            self.print_progress(t, elbo, sess)

    def print_progress(self, t, elbos, sess):
        if t % self.n_print == 0:
            print("iter %d elbo %.2f " % (t, np.mean(elbos)))
            self.q.print_params(sess)

    def build_score_loss(self):
        pass

    def build_reparam_loss(self):
        pass

class MFVI(VI):
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
        if hasattr(self.q, 'entropy'):
            # ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
            # where entropy is analytic
            q_entropy = self.q.entropy()
            self.elbos = self.model.log_prob(z) + q_entropy
            return tf.reduce_mean(q_log_prob * \
                                  tf.stop_gradient(self.log_prob(z))) + \
                   q_entropy
        else:
            # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.q.num_vars):
                q_log_prob += self.q.log_prob_zi(i, self.samples)

            self.elbos = self.model.log_prob(self.samples) - q_log_prob
            return -tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbos))

    def build_reparam_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        """
        z = self.q.reparam(self.samples)

        if hasattr(self.q, 'entropy'):
            # ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
            # where entropy is analytic
            self.elbos = tf.reduce_mean(self.model.log_prob(z)) + \
                        self.q.entropy()
        else:
            # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
            q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
            for i in range(self.q.num_vars):
                q_log_prob += self.q.log_prob_zi(i, z)

            self.elbos = tf.reduce_mean(self.model.log_prob(z) - q_log_prob)

        return -self.elbos

class AlphaVI(VI):
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
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, self.samples)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        log_omega = self.model.log_prob(self.samples) - q_log_prob
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
        z = self.q.reparam(self.samples)
        # ELBO = E_{q(z; lambda)} [ w(z; lambda)^{1-alpha} ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, z)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        log_omega = self.model.log_prob(z) - q_log_prob
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
            self.q.print_params(sess)

class LiVI(VI):
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
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, self.samples)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        log_omega = self.model.log_prob(self.samples) - q_log_prob
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
        z = self.q.reparam(self.samples)
        # ELBO = E_{q(z; lambda)} [ log w(z; lambda)^{1-alpha} ]
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, z)

        # 1/B sum_{b=1}^B exp{ log(omega) }
        # = exp{ max_log_omega } *
        #   (1/B sum_{b=1}^B exp{ log(omega) - max_log_omega})
        log_omega = self.model.log_prob(self.samples) - q_log_prob
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
