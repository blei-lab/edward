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
        #self.lamda = tf.Variable(tf.sigmoid(tf.random_normal([num_vars])))
        self.lamda = tf.Variable(tf.random_normal([num_vars]))

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        lamda_const = sess.run([tf.sigmoid(self.lamda)])[0]
        # TODO generalize to higher dimensions
        return bernoulli.rvs(lamda_const, size=size)

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        return bernoulli_log_prob(z[:, i], tf.sigmoid(self.lamda[i]))

class MFBeta:
    """
    q(z | lambda ) = prod_{i=1}^d Beta(z[i] | lambda[i])
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = 2*num_vars
        self.alpha = tf.Variable(tf.nn.softplus(tf.random_normal([num_vars])))
        self.beta = tf.Variable(tf.nn.softplus(tf.random_normal([num_vars])))
        # TODO make all variables outside, not in these classes but as
        # part of inference most generally

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([self.alpha, self.beta])
        # TODO generalize to higher dimensions
        return beta.rvs(a, b, size=size)

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        # TODO generalize to higher dimensions
        #return beta_log_prob(z[:, i], self.alpha[i], self.beta[i])
        return beta_log_prob(z[:, i], self.alpha, self.beta)
