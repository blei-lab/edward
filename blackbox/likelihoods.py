import numpy as np
import tensorflow as tf
from scipy.stats import bernoulli, beta
from dists import bernoulli_log_prob, beta_log_prob

class MFBernoulli:
    """
    q(z | lambda ) = prod_{i=1}^d Bernoulli(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = num_vars
        # TODO something about constraining the parameters in simplex
        # TODO we're storing unconstrained here. be consistent
        self.lamda = tf.Variable(tf.random_normal([num_vars]))

    # TODO use __str__(self):
    def print_params(self, sess):
        lamda_const = sess.run([tf.sigmoid(self.lamda)])[0]
        if lamda_const.size > 1:
            lamda_const[-1] = 1.0 - np.sum(lamda_const[:-1])

        print "p:"
        print lamda_const

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        lamda_const = sess.run([tf.sigmoid(self.lamda)])[0]
        if lamda_const.size > 1:
            lamda_const[-1] = 1.0 - np.sum(lamda_const[:-1])

        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = bernoulli.rvs(lamda_const[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        if i < self.num_vars:
            return bernoulli_log_prob(z[:, i], tf.sigmoid(self.lamda[i]))
        else:
            return bernoulli_log_prob(z[:, i],
                1.0 - tf.reduce_sum(tf.sigmoid(self.lamda[-1])))

class MFBeta:
    """
    q(z | lambda ) = prod_{i=1}^d Beta(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = 2*num_vars
        # TODO apply transforms not here but whenever using the params
        self.alpha = tf.Variable(tf.nn.softplus(tf.random_normal([num_vars])))
        self.beta = tf.Variable(tf.nn.softplus(tf.random_normal([num_vars])))
        # TODO make all variables outside, not in these classes but as
        # part of inference most generally

    # TODO use __str__(self):
    def print_params(self, sess):
        a, b = sess.run([self.alpha, self.beta])

        print "alpha:"
        print a
        print "beta:"
        print b

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([self.alpha, self.beta])
        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = beta.rvs(a[d], b[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        return beta_log_prob(z[:, i], self.alpha[i], self.beta[i])
