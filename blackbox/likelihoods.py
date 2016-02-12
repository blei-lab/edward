import tensorflow as tf
from scipy.stats import beta
from dists import bernoulli_log_prob, beta_log_prob

class MFBernoulli:
    """
    q(z | lambda ) = prod_{i=1}^d Bernoulli(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = num_vars
        self.lamda = tf.Variable(tf.random_uniform([num_vars]))

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        return bernoulli_log_prob(z[i], self.lamda[i])

class MFBeta:
    """
    q(z | lambda ) = prod_{i=1}^d Beta(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = 2*num_vars
        # TODO setting seed is hard
        #self.alpha = tf.Variable(tf.random_uniform([num_vars]))
        #self.beta = tf.Variable(tf.random_uniform([num_vars]))
        self.alpha = tf.Variable(1.0)
        self.beta = tf.Variable(2.0)

    def sample(self, size, a, b):
        """z ~ q(z | lambda)"""
        # TODO generalize to higher dimensions
        return beta.rvs(a, b, size=size)

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        # TODO generalize to higher dimensions
        #return beta_log_prob(z[i], self.alpha[i], self.beta[i])
        return beta_log_prob(z, self.alpha, self.beta)
