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
    raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class AB_divergence(VariationalInference):
    """Variational inference with the AB divergence

    $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

    This class minimizes the objective by automatically selecting from a
    variety of black box inference techniques.

    #### Notes

    `KLqp` also optimizes any model parameters $p(z \mid x;
    \\theta)$. It does this by variational EM, minimizing

    $\mathbb{E}_{q(z; \lambda)} [ \log p(x, z; \\theta) ]$

    with respect to $\\theta$.

    In conditional inference, we infer $z$ in $p(z, \\beta
    \mid x)$ while fixing inference over $\\beta$ using another
    distribution $q(\\beta)$. During gradient calculation, instead
    of using the model's density

    $\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

    for each sample $s=1,\ldots,S$, `KLqp` uses

    $\log p(x, z^{(s)}, \\beta^{(s)}),$

    where $z^{(s)} \sim q(z; \lambda)$ and $\\beta^{(s)}
    \sim q(\\beta)$.
    """

    def __init__(self, *args, **kwargs):
        super(AB_divergence, self).__init__(*args, **kwargs)

    def initialize(self, n_samples=32, kl_scaling=None,
                   alpha=.2, beta=0.5, batch_size=32, *args, **kwargs):
        """Initialize inference algorithm. It initializes hyperparameters
        and builds ops for the algorithm's computation graph.

        Args:
          n_samples: int, optional.
            Number of samples from variational model for calculating
            stochastic gradients.
          kl_scaling: dict of RandomVariable to float, optional.
            Provides option to scale terms when using ELBO with KL divergence.
            If the KL divergence terms are

            $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
                  \log q(z\mid x, \lambda) - \log p(z)],$

            then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
            where $\\alpha_p$ is a float that specifies how much to
            scale the KL term.
        """
        # print("+++ in initialize ab_div")
        # print("+++ n_samples = {}".format(n_samples))

        if kl_scaling is None:
            kl_scaling = {}

        self.n_samples = n_samples
        self.kl_scaling = kl_scaling
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        return super(AB_divergence, self).initialize(*args, **kwargs)

    def build_loss_and_gradients(self, var_list):
        """Wrapper for the `ac_divergence` loss function.

        TODO
        """
        is_reparameterizable = all([
                                       rv.reparameterization_type ==
                                       tf.contrib.distributions.FULLY_REPARAMETERIZED
                                       for rv in six.itervalues(self.latent_vars)])
        is_analytic_kl = all([isinstance(z, Normal) and isinstance(qz, Normal)
                              for z, qz in six.iteritems(self.latent_vars)])
        if not is_analytic_kl and self.kl_scaling:
            # raise NotImplementedError("non analytic KL not implemented yet")
            raise TypeError("kl_scaling must be None when using non-analytic KL term")
        if is_reparameterizable:
            if is_analytic_kl:
                # See function ### 1 ###
                return build_reparam_loss_and_gradients(self, var_list,
                                                        alpha=self.alpha, beta=self.beta,
                                                        batch_size=self.batch_size)
                # return build_reparam_ab_loss_and_gradients(self, var_list,
                #                                            alpha=self.alpha, beta=self.beta)
            else:
                # See function ### 2 ###
                return build_reparam_loss_and_gradients(self, var_list,
                                                        alpha=self.alpha, beta=self.beta,
                                                        batch_size=self.batch_size)
        else:
            if is_analytic_kl:
                # See function ### 3 ###
                return build_score_ab_loss_and_gradients(self, var_list,
                                                         alpha=self.alpha, beta=self.beta)
            else:
                # See function ### 4 ###
                return build_score_loss_and_gradients(self, var_list,
                                                      alpha=self.alpha, beta=self.beta, batch_size=self.batch_size)


# Function ### 1 ###
def build_reparam_ab_loss_and_gradients(inference, var_list, alpha, beta):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -\\text{ELBO} =  - ( \mathbb{E}_{q(z; \lambda)} [ \log p(x \mid z) ]
            + \\text{KL}(q(z; \lambda) \| p(z)) )

    based on the reparameterization trick (Kingma and Welling, 2014).

    It assumes the AB-divergence is analytic.

    Computed by sampling from $q(z;\lambda)$ and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_lik = [0.0] * inference.n_samples
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    for s in range(inference.n_samples):
        # Form dictionary in order to replace conditioning on prior or
        # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(inference.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(inference.latent_vars):
            # Copy q(z) to obtain new set of posterior samples.
            qz_copy = copy(qz, scope=scope)
            dict_swap[z] = qz_copy.value()

        for x in six.iterkeys(inference.data):
            if isinstance(x, RandomVariable):
                x_copy = copy(x, dict_swap, scope=scope)
                p_log_lik[s] += tf.reduce_sum(
                    inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

    p_log_lik = tf.reduce_mean(p_log_lik)

    kl_penalty = tf.reduce_sum([
                                   inference.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
                                   for z, qz in six.iteritems(inference.latent_vars)])

    if inference.logging:
        tf.summary.scalar("loss/p_log_lik", p_log_lik,
                          collections=[inference._summary_key])
        tf.summary.scalar("loss/kl_penalty", kl_penalty,
                          collections=[inference._summary_key])

    loss = -(p_log_lik - kl_penalty)

    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars


# See function ### 2 ###
def build_reparam_loss_and_gradients(inference, var_list, alpha=1.0, beta=0.0, batch_size=32):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    TODO

    $-\\text{ELBO} =
        -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

    based on the reparameterization trick (Kingma and Welling, 2014).

    Computed by sampling from $q(z;\lambda)$ and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_prob = [0.0] * inference.n_samples
    q_log_prob = [0.0] * inference.n_samples
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    for s in range(inference.n_samples):
        # Form dictionary in order to replace conditioning on prior or
        # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(inference.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(inference.latent_vars):
            # Copy q(z) to obtain new set of posterior samples.
            qz_copy = copy(qz, scope=scope)
            dict_swap[z] = qz_copy.value()
            q_log_prob[s] += tf.reduce_sum(
                inference.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

        # print("q_log_prob= {}".format(q_log_prob))

        for z in six.iterkeys(inference.latent_vars):
            z_copy = copy(z, dict_swap, scope=scope)
            p_log_prob[s] += tf.reduce_sum(
                inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

        for x in six.iterkeys(inference.data):
            if isinstance(x, RandomVariable):
                x_copy = copy(x, dict_swap, scope=scope)
                p_log_prob[s] += tf.reduce_sum(
                    inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
        # print("p_log_prob= {}".format(p_log_prob))

    # Reduces to a Renyi divergence:
    if isclose(alpha + beta, 1.0, abs_tol=1e-4):
        print("special case")

        logF = [p - q for p, q in zip(p_log_prob, q_log_prob)]

        if np.abs(alpha - 1.0) < 10e-3:
            divergence = tf.reduce_mean(logF)
        else:
            logF = tf.reshape(logF, [inference.n_samples, 1])
            logF = logF * (1 - alpha)
            logF_max = tf.reduce_max(logF, 0)
            logF = tf.log(tf.clip_by_value(tf.reduce_mean(tf.exp(logF - logF_max), 0), 1e-9, np.inf))
            logF = (logF + logF_max) / (1 - alpha)
            loss = tf.reduce_mean(logF)

        divergence = -divergence

    # AB-objective:
    else:
        logF1 = tf.stack([(alpha + beta - 1) * q for q in q_log_prob])
        logF2 = [(alpha + beta) * p - q for p, q in zip(p_log_prob, q_log_prob)]
        logF3 = [beta * p - (1 - alpha) * q for p, q in zip(p_log_prob, q_log_prob)]

        # print("q_log_prob= {}".format(q_log_prob))
        # print("logF1= {}".format(logF1))

        # TODO Wrong should be n_samples, batch_size
        logF1 = tf.reshape(logF1, [inference.n_samples, 1])
        logF2 = tf.reshape(logF2, [inference.n_samples, 1])
        logF3 = tf.reshape(logF3, [inference.n_samples, 1])

        # print("logF1= {}".format(logF1))

        logF1_max = tf.reduce_max(logF1, 0)
        logF2_max = tf.reduce_max(logF2, 0)
        logF3_max = tf.reduce_max(logF3, 0)

        logF1 = tf.log(tf.reduce_mean(tf.exp(logF1 - logF1_max), 0))
        logF2 = tf.log(tf.reduce_mean(tf.exp(logF2 - logF2_max), 0))
        logF3 = tf.log(tf.reduce_mean(tf.exp(logF3 - logF3_max), 0))

        logF = (logF1 + logF1_max) / (beta * (alpha + beta)) \
               + (logF2 + logF2_max) / (alpha * (alpha + beta)) \
               - (logF3 + logF3_max) / (alpha * beta)

        logF = tf.clip_by_value(logF, 0, np.inf)

        loss = tf.reduce_mean(logF)

    if inference.logging:
        p_log_prob = tf.reduce_mean(p_log_prob)
        q_log_prob = tf.reduce_mean(q_log_prob)
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[inference._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[inference._summary_key])

    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars


#############
### UTILS ###
#############
def isclose(a, b, rel_tol=1e-06, abs_tol=0.0):
    r"""
    Almost equal

    :param a:
    :param b:
    :param rel_tol:
    :param abs_tol:
    :return:
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
