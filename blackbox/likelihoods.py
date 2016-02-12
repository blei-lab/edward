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
        self.p_unconst = tf.Variable(tf.random_normal([num_vars]))

    # TODO use __str__(self):
    def print_params(self, sess):
        p = sess.run([tf.sigmoid(self.p_unconst)])[0]
        if p.size > 1:
            p[-1] = 1.0 - np.sum(p[:-1])

        print "p:"
        print p

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        p = sess.run([tf.sigmoid(self.p_unconst)])[0]
        if p.size > 1:
            p[-1] = 1.0 - np.sum(p[:-1])

        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = bernoulli.rvs(p[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        if i < self.num_vars:
            pi = tf.sigmoid(self.p_unconst[i])
        else:
            pi = 1.0 - tf.reduce_sum(tf.sigmoid(self.p_unconst[-1]))

        return bernoulli_log_prob(z[:, i], pi)

class MFBeta:
    """
    q(z | lambda ) = prod_{i=1}^d Beta(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = 2*num_vars
        # TODO apply transforms not here but whenever using the params
        self.a_unconst = tf.Variable(tf.random_normal([num_vars]))
        self.b_unconst = tf.Variable(tf.random_normal([num_vars]))
        # TODO make all variables outside, not in these classes but as
        # part of inference most generally

    # TODO use __str__(self):
    def print_params(self, sess):
        a, b = sess.run([ \
            tf.nn.softplus(self.a_unconst),
            tf.nn.softplus(self.b_unconst)])

        print "alpha:"
        print a
        print "beta:"
        print b

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([ \
            tf.nn.softplus(self.a_unconst),
            tf.nn.softplus(self.b_unconst)])

        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = beta.rvs(a[d], b[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        ai = tf.nn.softplus(self.a_unconst)[i]
        bi = tf.nn.softplus(self.b_unconst)[i]
        # TODO
        #ai = tf.nn.softplus(self.a_unconst[i])
        #bi = tf.nn.softplus(self.b_unconst[i])
        return beta_log_prob(z[:, i], ai, bi)

#class MFGaussian:
