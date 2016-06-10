from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.data import Data
from edward.models import Variational, PointMass
from edward.util import kl_multivariate_normal, log_sum_exp, get_session

try:
    import prettytensor as pt
except ImportError:
    pass

class Inference:
    """
    Base class for inference methods.

    Parameters
    ----------
    model : Model
        probability model p(x, z)
    data : Data, optional
        data x
    """
    def __init__(self, model, data=Data()):
        self.model = model
        self.data = data
        get_session()


class MonteCarlo(Inference):
    """
    Base class for Monte Carlo methods.

    Parameters
    ----------
    model : Model
        probability model p(x, z)
    data : Data, optional
        data x
    """
    def __init__(self, *args, **kwargs):
        Inference.__init__(self, *args, **kwargs)

class VariationalInference(Inference):
    """
    Base class for variational inference methods.

    Parameters
    ----------
    model : Model
        probability model p(x, z)
    variational : Variational
        variational model q(z; lambda)
    data : Data, optional
        data x
    """
    def __init__(self, model, variational, data=Data()):
        Inference.__init__(self, model, data)
        self.variational = variational

    def run(self, *args, **kwargs):
        """
        A simple wrapper to run the inference algorithm.
        """
        self.initialize(*args, **kwargs)
        for t in range(self.n_iter):
            loss = self.update()
            self.print_progress(t, loss)


    def initialize(self, n_iter=1000, n_data=None, n_print=100,
        optimizer=None):
        """
        Initialize inference algorithm.

        Parameters
        ----------
        n_iter : int, optional
            Number of iterations for optimization.
        n_data : int, optional
            Number of samples for data subsampling. Default is to use all
            the data.
        n_print : int, optional
            Number of iterations for each print progress.
        optimizer : str, optional
            Whether to use TensorFlow optimizer or PrettyTensor
            optimizer if using PrettyTensor. Defaults to TensorFlow.
        """
        self.n_iter = n_iter
        self.n_data = n_data
        self.n_print = n_print

        self.loss = tf.constant(0.0)

        loss = self.build_loss()
        if optimizer == None:
            # Use ADAM with a decaying scale factor
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                global_step,
                                                100, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = optimizer.minimize(loss, global_step=global_step)
        else:
            optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
            self.train = pt.apply_optimizer(optimizer, losses=[loss])

        init = tf.initialize_all_variables()
        init.run()

    def update(self):
        self.train.run()
        loss = self.loss.eval()
        return loss

    def print_progress(self, t, loss):
        if t % self.n_print == 0:
            print("iter {:d} loss {:.2f}".format(t, loss))
            self.variational.print_params()

    def build_loss(self):
        raise NotImplementedError()

    def test(self,sess):
        x = self.data.test_data
        z, self.samples = self.variational.sample(x)
    
        eval_metric = getattr(self.model, 'evaluation_metric', None)  
        if eval_metric is None:
             eval_metric = getattr(self.model, 'log_prob', None)  

        loss = sess.run([eval_metric(x, z)])
        return loss[0][0]

class MFVI(VariationalInference):
# TODO this isn't MFVI so much as VI where q is analytic
    """
    Mean-field variational inference
    (Ranganath et al., 2014)
    """
    def __init__(self, *args, **kwargs):
        VariationalInference.__init__(self, *args, **kwargs)

    def initialize(self, n_minibatch=1, score=None, *args, **kwargs):
        """
        Parameters
        ----------
        n_minibatch : int, optional
            Number of samples from variational model for calculating
            stochastic gradients.
        score : bool, optional
            Whether to force inference to use the score function
            gradient estimator. Otherwise default is to use the
            reparameterization gradient if available.
        """
        if score is None and self.variational.is_reparam:
            self.score = False
        else:
            self.score = True

        self.n_minibatch = n_minibatch
        return VariationalInference.initialize(self, *args, **kwargs)

    def update(self):
        sess = get_session()
        feed_dict = self.variational.np_sample(self.samples, self.n_minibatch)
        _, loss = sess.run([self.train, self.loss], feed_dict)
        return loss

    def build_loss(self):
        if self.score:
            if self.variational.is_normal and hasattr(self.model, 'log_lik'):
                return self.build_score_loss_kl()
            # Analytic entropies may lead to problems around
            # convergence; for now it is deactivated.
            #elif self.variational.is_entropy:
            #    return self.build_score_loss_entropy()
            else:
                return self.build_score_loss()
        else:
            if self.variational.is_normal and hasattr(self.model, 'log_lik'):
                return self.build_reparam_loss_kl()
            #elif self.variational.is_entropy:
            #    return self.build_reparam_loss_entropy()
            else:
                return self.build_reparam_loss()

    def build_score_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.
        (Paisley et al., 2012)

        ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
        """
        x = self.data.sample(self.n_data)
        z, self.samples = self.variational.sample(x, self.n_minibatch)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, tf.stop_gradient(z))

        losses = self.model.log_prob(x, z) - q_log_prob
        self.loss = tf.reduce_mean(losses)
        return -tf.reduce_mean(q_log_prob * tf.stop_gradient(losses))

    def build_reparam_loss(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.
        (Kingma and Welling, 2014)

        ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
        """
        x = self.data.sample(self.n_data)
        z, self.samples = self.variational.sample(x, self.n_minibatch)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, z)

        self.loss = tf.reduce_mean(self.model.log_prob(x, z) - q_log_prob)
        return -self.loss

    def build_score_loss_kl(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.

        ELBO = E_{q(z; lambda)} [ log p(x | z) ] + KL(q(z; lambda) || p(z))
        where KL is analytic

        It assumes the model prior is p(z) = N(z; 0, 1).
        """
        x = self.data.sample(self.n_data)
        z, self.samples = self.variational.sample(x, self.n_minibatch)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, tf.stop_gradient(z))

        p_log_lik = self.model.log_lik(x, z)
        mu = tf.pack([layer.m for layer in self.variational.layers])
        sigma = tf.pack([layer.s for layer in self.variational.layers])
        kl = kl_multivariate_normal(mu, sigma)
        self.loss = tf.reduce_mean(p_log_lik) - kl
        return -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_lik)) - kl)

    def build_score_loss_entropy(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the score function estimator.

        ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
        where entropy is analytic
        """
        x = self.data.sample(self.n_data)
        z, self.samples = self.variational.sample(x, self.n_minibatch)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, tf.stop_gradient(z))

        p_log_prob = self.model.log_prob(x, z)
        q_entropy = self.variational.entropy()
        self.loss = tf.reduce_mean(p_log_prob) + q_entropy
        return -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) +
                 q_entropy)

    def build_reparam_loss_kl(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.

        ELBO = E_{q(z; lambda)} [ log p(x | z) ] + KL(q(z; lambda) || p(z))
        where KL is analytic

        It assumes the model prior is p(z) = N(z; 0, 1).
        """
        x = self.data.sample(self.n_data)
        z, self.samples = self.variational.sample(x, self.n_minibatch)

        mu = tf.pack([layer.m for layer in self.variational.layers])
        sigma = tf.pack([layer.s for layer in self.variational.layers])
        self.loss = tf.reduce_mean(self.model.log_lik(x, z)) - \
                    kl_multivariate_normal(mu, sigma)
        return -self.loss

    def build_reparam_loss_entropy(self):
        """
        Loss function to minimize, whose gradient is a stochastic
        gradient based on the reparameterization trick.

        ELBO = E_{q(z; lambda)} [ log p(x, z) ] + H(q(z; lambda))
        where entropy is analytic
        """
        x = self.data.sample(self.n_data)
        z, self.samples = self.variational.sample(x, self.n_minibatch)
        self.loss = tf.reduce_mean(self.model.log_prob(x, z)) + \
                    self.variational.entropy()
        return -self.loss

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
        return VariationalInference.initialize(self, *args, **kwargs)

    def update(self):
        sess = get_session()
        feed_dict = self.variational.np_sample(self.samples, self.n_minibatch)
        _, loss = sess.run([self.train, self.loss], feed_dict)
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
        z, self.samples = self.variational.sample(x, self.n_minibatch)

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.variational.num_factors):
            q_log_prob += self.variational.log_prob_zi(i, z)

        # 1/B sum_{b=1}^B grad_log_q * w_norm
        # = 1/B sum_{b=1}^B grad_log_q * exp{ log(w_norm) }
        log_w = self.model.log_prob(x, z) - q_log_prob

        # normalized log importance weights
        log_w_norm = log_w - log_sum_exp(log_w)
        w_norm = tf.exp(log_w_norm)

        self.loss = tf.reduce_mean(w_norm * log_w)
        return -tf.reduce_mean(q_log_prob * tf.stop_gradient(w_norm))

class MAP(VariationalInference):
    """
    Maximum a posteriori
    """
    def __init__(self, model, data=Data(), transform=tf.identity):
        if hasattr(model, 'num_vars'):
            variational = Variational()
            variational.add(PointMass(model.num_vars, transform))
        else:
            variational = Variational()
            variational.add(PointMass(0, transform))

        VariationalInference.__init__(self, model, variational, data)

    def build_loss(self):
        x = self.data.sample(self.n_data)
        z, _ = self.variational.sample(x)
        self.loss = tf.squeeze(self.model.log_prob(x, z))
        return -self.loss
