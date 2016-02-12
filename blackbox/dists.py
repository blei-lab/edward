import numpy as np
import tensorflow as tf

from util import log_beta

def bernoulli_log_prob(x, p):
    """
    log Bernoulli(x; p)
    """
    clip_finite = True
    if clip_finite:
        # avoid taking log(0) for float32 inputs
        # TODO: adapt to float64, etc.
        p = tf.clip_by_value(p, 1e-45, 1.0)

    lp = tf.log(p)
    lp1 = tf.log(1.0-p)
    return tf.mul(x, lp) + tf.mul(1-x, lp1)

def beta_log_prob(x, alpha=1.0, beta=1.0):
    """
    log Beta(x; alpha, beta)
    """
    log_b = log_beta(alpha, beta)
    return (alpha - 1) * tf.log(x) + (beta-1) * tf.log(1-x) - log_b
