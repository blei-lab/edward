from __future__ import print_function
import numpy as np
import tensorflow as tf

from scipy.stats import bernoulli, beta, norm
from blackbox.stats import bernoulli_log_prob, beta_log_prob, gaussian_log_prob, gaussian_entropy
from blackbox.util import get_dims

class MFBernoulli:
    """
    q(z | lambda ) = prod_{i=1}^d Bernoulli(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = num_vars

        self.p_unconst = tf.Variable(tf.random_normal([num_vars]))
        # TODO make all variables outside, not in these classes but as
        # part of inference most generally
        self.transform = tf.sigmoid
        # TODO something about constraining the parameters in simplex
        # TODO deal with truncations

    # TODO use __str__(self):
    def print_params(self, sess):
        p = sess.run([self.transform(self.p_unconst)])[0]
        if p.size > 1:
            p[-1] = 1.0 - np.sum(p[:-1])

        print("probability:")
        print(p)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        p = sess.run([self.transform(self.p_unconst)])[0]
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
            pi = self.transform(self.p_unconst[i])
        else:
            pi = 1.0 - tf.reduce_sum(self.transform(self.p_unconst[-1]))

        return bernoulli_log_prob(z[:, i], pi)

class MFBeta:
    """
    q(z | lambda ) = prod_{i=1}^d Beta(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = 2*num_vars

        self.a_unconst = tf.Variable(tf.random_normal([num_vars]))
        self.b_unconst = tf.Variable(tf.random_normal([num_vars]))
        self.transform = tf.nn.softplus

    def print_params(self, sess):
        a, b = sess.run([ \
            self.transform(self.a_unconst),
            self.transform(self.b_unconst)])

        print("shape:")
        print(a)
        print("scale:")
        print(b)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([ \
            self.transform(self.a_unconst),
            self.transform(self.b_unconst)])

        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = beta.rvs(a[d], b[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        ai = self.transform(self.a_unconst)[i]
        bi = self.transform(self.b_unconst)[i]
        # TODO
        #ai = self.transform(self.a_unconst[i])
        #bi = self.transform(self.b_unconst[i])
        return beta_log_prob(z[:, i], ai, bi)

class MFGaussian:
    """
    q(z | lambda ) = prod_{i=1}^d Gaussian(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = 2*num_vars

        self.m_unconst = tf.Variable(tf.random_normal([num_vars]))
        self.s_unconst = tf.Variable(tf.random_normal([num_vars]))
        self.transform_m = tf.identity
        self.transform_s = tf.nn.softplus

    def print_params(self, sess):
        m, s = sess.run([ \
            self.transform_m(self.m_unconst),
            self.transform_s(self.s_unconst)])

        print("mean:")
        print(m)
        print("std dev:")
        print(s)

    def sample_noise(self, size):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        # Not using this, since TensorFlow has a large overhead
        # whenever calling sess.run().
        #samples = sess.run(tf.random_normal(self.samples.get_shape()))
        return norm.rvs(size=size)

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        m = self.transform_m(self.m_unconst)
        s = self.transform_s(self.s_unconst)
        return m + eps * s

    def sample(self, size, sess):
        """
        z ~ q(z | lambda)
        """
        return sess.run(self.reparam(self.sample_noise(size)))

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        mi = self.transform_m(self.m_unconst)[i]
        si = self.transform_s(self.s_unconst)[i]
        # TODO
        #mi = self.transform_m(self.m_unconst[i])
        #si = self.transform_s(self.s_unconst[i])
        return tf.pack([gaussian_log_prob(zm[i], mi, si*si)
                        for zm in tf.unpack(z)])
        # TODO
        #return gaussian_log_prob(z[:, i], mi, si)

    # TODO entropy is bugged
    #def entropy(self):
    #    return gaussian_entropy(self.transform_s(self.s_unconst))
