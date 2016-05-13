from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.data import Data
from edward.util import kl_multivariate_normal, log_sum_exp
from edward.variationals import PointMass

try:
    import prettytensor as pt
except ImportError:
    pass

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

        return sess

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
    (Ranganath et al., 2014)
    """
    def __init__(self, *args, **kwargs):
        VariationalInference.__init__(self, *args, **kwargs)

    def initialize(self, n_minibatch=1, score=None, *args, **kwargs):
        # TODO if score=True, make Normal do sess.run()
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
        if score is None and self.variational.is_reparam:
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
            # TODO the mapping should go here before sampling.
            # In principle the mapping should go here but we don't
            # want to have to run this twice. Also I've noticed that it
            # is significantly slower if I have it here for some reason,
            # so I'm leaving this as an open problem.
            #x = self.data.sample(self.n_data)
            #self.variational.set_params(self.variational.mapping(x))
            samples = self.variational.sample(self.n_minibatch, sess)
        else:
            samples = self.variational.sample_noise(self.n_minibatch)

        _, loss = sess.run([self.train, self.losses], {self.samples: samples})

        return loss

    def build_loss(self):
        if self.score and hasattr(self.variational, 'entropy'):
            return self.build_score_loss_entropy()
        elif self.score:
            return self.build_score_loss()
        elif not self.score and hasattr(self.variational, 'entropy'):
            return self.build_reparam_loss_entropy()
        else:
            return self.build_reparam_loss()

    def build_score_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        (Paisley et al., 2012)
        """
        # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
        x = self.data.sample(self.n_data)
        self.variational.set_params(self.variational.mapping(x))

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, self.samples)

        self.losses = self.model.log_prob(x, self.samples) - q_log_prob
        return -tf.reduce_mean(q_log_prob * tf.stop_gradient(self.losses))

    def build_reparam_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        (Kingma and Welling, 2014)
        """
        # ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
        x = self.data.sample(self.n_data)
        self.variational.set_params(self.variational.mapping(x))
        z = self.variational.reparam(self.samples)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, z)

        self.losses = self.model.log_prob(x, z) - q_log_prob

        return -tf.reduce_mean(self.losses)

    def build_score_loss_entropy(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        """
        # ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
        # where entropy is analytic
        x = self.data.sample(self.n_data)
        self.variational.set_params(self.variational.mapping(x))

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, self.samples)

        x = self.data.sample(self.n_data)
        p_log_prob = self.model.log_prob(x, self.samples)
        q_entropy = self.variational.entropy()
        self.losses = p_log_prob + q_entropy
        return tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) + \
               q_entropy

    def build_reparam_loss_entropy(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        """
        # ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
        # where entropy is analytic
        x = self.data.sample(self.n_data)
        self.variational.set_params(self.variational.mapping(x))
        z = self.variational.reparam(self.samples)
        self.losses = self.model.log_prob(x, z) + self.variational.entropy()
        return -tf.reduce_mean(self.losses)

class VAE(VariationalInference):
    # TODO refactor into MFVI
    def __init__(self, *args, **kwargs):
        VariationalInference.__init__(self, *args, **kwargs)

    def initialize(self, n_data=None):
        # TODO refactor to use VariationalInference's initialize()
        self.n_data = n_data

        # TODO don't fix number of covariates
        self.x = tf.placeholder(tf.float32, [self.n_data, 28 * 28])
        self.losses = tf.constant(0.0)

        loss = self.build_loss()
        optimizer = tf.train.AdamOptimizer(1e-2, epsilon=1.0)
        # TODO move this to not rely on Pretty Tensor
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
        # In general, there should be a scale factor due to data
        # subsampling, so that
        # ELBO = N / M * ( ELBO using x_b )
        # where x^b is a mini-batch of x, with sizes M and N respectively.
        # This is absorbed into the learning rate.
        with tf.variable_scope("model") as scope:
            self.variational.set_params(self.variational.mapping(self.x))
            z = self.variational.sample(self.n_data)
            self.losses = tf.reduce_sum(self.model.log_likelihood(self.x, z)) - \
                          kl_multivariate_normal(self.variational.m,
                                                 self.variational.s)

        return -self.losses

class KLpq(VariationalInference):
    """
    Kullback-Leibler divergence from posterior to variational model,
    KL( p(z |x) || q(z) ).
    (Cappe et al., 2008)
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
        samples = self.variational.sample(self.n_minibatch, sess)
        _, loss = sess.run([self.train, self.losses], {self.samples: samples})
        return loss

    def build_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient inspired by adaptive importance sampling.
        """
        # loss = E_{q(z; lambda)} [ w_norm(z; lambda) *
        #                           ( log p(x, z) - log q(z; lambda) ) ]
        # where
        # w_norm(z; lambda) = w(z; lambda) / sum_z( w(z; lambda) )
        # w(z; lambda) = p(x, z) / q(z; lambda)
        #
        # gradient = - E_{q(z; lambda)} [ w_norm(z; lambda) *
        #                                 grad_{lambda} log q(z; lambda) ]
        x = self.data.sample(self.n_data)
        self.variational.set_params(self.variational.mapping(x))

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, self.samples)

        # 1/B sum_{b=1}^B grad_log_q * w_norm
        # = 1/B sum_{b=1}^B grad_log_q * exp{ log(w_norm) }
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
    def __init__(self, model, data=Data(), transform=tf.identity):
        variational = PointMass(model.num_vars, transform)
        VariationalInference.__init__(self, model, variational, data)

    def build_loss(self):
        x = self.data.sample(self.n_data)
        self.variational.set_params(self.variational.mapping(x))
        z = self.variational.get_params()
        self.losses = self.model.log_prob(x, z)
        return -tf.reduce_mean(self.losses)
