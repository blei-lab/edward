from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy

try:
    from edward.models import Normal
    from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
    raise ImportError(
        "{0}. Your TensorFlow version is not supported.".format(e))


class Renyi_divergence(VariationalInference):
    """Variational inference with the Renyi divergence

    $ \text{D}_{R}^{(\alpha)}(q(z)||p(z \mid x))
        = \frac{1}{\alpha-1} \log \int q(z)^{\alpha} p(z \mid x)^{1-\alpha} dz $

    To perform the optimization, this class uses the techniques from
    Renyi Divergence Variational Inference (Y. Li & al, 2016)

    # Notes:
        - Renyi divergence does not have any analytic version.
        - Renyi divergence does not have any version for non reparametrizable
            models.
        - backward_pass = 'max': (extreme case $\alpha \rightarrow -\infty$)
            the algorithm chooses the sample that has the maximum unnormalised
            importance weight. This does not minimize the Renyi divergence
            anymore.
        - backward_pass = 'min': (extreme case $\alpha \rightarrow +\infty$)
            the algorithm chooses the sample that has the minimum unnormalised
            importance weight. This does not minimize the Renyi divergence
            anymore. This mode is not describe in the paper but implemented
            in the publicly available implementation of the paper's experiments.
    """
    def __init__(self, *args, **kwargs):
        super(Renyi_divergence, self).__init__(*args, **kwargs)

    def initialize(self,
                   n_samples=32,
                   alpha=1.,
                   backward_pass='full',
                   *args, **kwargs):
        """Initialize inference algorithm. It initializes hyperparameters
        and builds ops for the algorithm's computation graph.

        Args:
            n_samples: int, optional.
                Number of samples from variational model for calculating
                stochastic gradients.
            alpha: float, optional.
                Renyi divergence coefficient.
            backward_pass: str, optional.
                Backward pass mode to be used.
                Options: 'min', 'max', 'full'
                (see Renyi Divergence Variational Inference (Y. Li & al, 2016)
                 section 4.2)
        """
        self.n_samples = n_samples
        self.alpha = alpha
        self.backward_pass = backward_pass

        return super(Renyi_divergence, self).initialize(*args, **kwargs)

    def build_loss_and_gradients(self, var_list):
        """Build the Renyi ELBO function.

        Its automatic differentiation is a stochastic gradient of

        $ \mcalL_{R}^{\alpha}(q; x) =
                \frac{1}{1-\alpha} \log \dsE_{q} \left[
                    \left( \frac{p(x, z)}{q(z)}\right)^{1-\alpha} \right] $

        It uses:
        1. Monte Carlo approximation of the ELBO (Y. Li & al, 2016)
        2. Reparameterization gradients (Kingma & al, 2014)
        3. Stochastic approximation of the joint distribution (Y. Li & al, 2016)

        # Notes
            If the model is not reparameterizable, it returns a
            NotImplementedError.
            See Renyi Divergence Variational Inference (Y. Li & al, 2016)
            for more details.
        """
        is_reparameterizable = all([
            rv.reparameterization_type ==
            tf.contrib.distributions.FULLY_REPARAMETERIZED
            for rv in six.itervalues(self.latent_vars)])

        if is_reparameterizable:
            p_log_prob = [0.0] * self.n_samples
            q_log_prob = [0.0] * self.n_samples
            base_scope = tf.get_default_graph().unique_name("inference") + '/'
            for s in range(self.n_samples):
                # Form dictionary in order to replace conditioning on prior or
                # observed variable with conditioning on a specific value.
                scope = base_scope + tf.get_default_graph().unique_name("sample")
                dict_swap = {}
                for x, qx in six.iteritems(self.data):
                    if isinstance(x, RandomVariable):
                        if isinstance(qx, RandomVariable):
                            qx_copy = copy(qx, scope=scope)
                            dict_swap[x] = qx_copy.value()
                        else:
                            dict_swap[x] = qx

                for z, qz in six.iteritems(self.latent_vars):
                    # Copy q(z) to obtain new set of posterior samples.
                    qz_copy = copy(qz, scope=scope)
                    dict_swap[z] = qz_copy.value()
                    q_log_prob[s] += tf.reduce_sum(
                        self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

                for z in six.iterkeys(self.latent_vars):
                    z_copy = copy(z, dict_swap, scope=scope)
                    p_log_prob[s] += tf.reduce_sum(
                        self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

                for x in six.iterkeys(self.data):
                    if isinstance(x, RandomVariable):
                        x_copy = copy(x, dict_swap, scope=scope)
                        p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

            logF = [p - q for p, q in zip(p_log_prob, q_log_prob)]

            if self.backward_pass == 'max':
                logF = tf.stack(logF)
                logF = tf.reduce_max(logF, 0)
                loss = tf.reduce_mean(logF)
            elif self.backward_pass == 'min':
                logF = tf.stack(logF)
                logF = tf.reduce_min(logF, 0)
                loss = tf.reduce_mean(logF)
            elif isclose(self.alpha, 1.0, abs_tol=10e-3):
                loss = tf.reduce_mean(logF)
            else:
                logF = tf.stack(logF)
                logF = logF * (1 - self.alpha)
                logF_max = tf.reduce_max(logF, 0)
                logF = tf.log(
                    tf.maximum(1e-9,
                               tf.reduce_mean(tf.exp(logF - logF_max), 0)))
                logF=(logF + logF_max) / (1 - self.alpha)
                loss=tf.reduce_mean(logF)
            loss=-loss

            if self.logging:
                p_log_prob=tf.reduce_mean(p_log_prob)
                q_log_prob=tf.reduce_mean(q_log_prob)
                tf.summary.scalar("loss/p_log_prob", p_log_prob,
                                  collections=[self._summary_key])
                tf.summary.scalar("loss/q_log_prob", q_log_prob,
                                  collections=[self._summary_key])

            grads=tf.gradients(loss, var_list)
            grads_and_vars=list(zip(grads, var_list))
            return loss, grads_and_vars
        else:
            raise NotImplementedError("Variational Renyi inference only works with reparameterizable models")


#############
### UTILS ###
#############
def isclose(a, b, rel_tol=0.0, abs_tol=1e-3):
    r"""
    Almost equal

    :param a:
    :param b:
    :param rel_tol:
    :param abs_tol:
    :return: Bool
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
