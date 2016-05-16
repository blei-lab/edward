from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import multinomial
from itertools import product
from scipy.special import gammaln

sess = tf.Session()

def multinomial_logpmf(x, n, p):
    """
    log pmf of multinomial. SciPy doesn't have it.

    Parameters
    ----------
    x: np.array
        vector of length K, where x[i] is the number of outcomes
        in the ith bucket
    n: int
        number of outcomes equal to sum x[i]
    p: np.array
        vector of probabilities summing to 1
    """
    return gammaln(n + 1.0) - \
           np.sum(gammaln(x + 1.0)) + \
           np.sum(x * np.log(p))

def multinomial_entropy(n, p):
    """
    entropy of multinomial. SciPy doesn't have it.

    Parameters
    ----------
    n: int
        number of outcomes equal to sum x[i]
    p: np.array
        vector of probabilities summing to 1
    """
    k = len(p)
    max_range = np.zeros(k, dtype=np.int32) + n
    x = np.array([i for i in product(*(range(i+1) for i in max_range))
                         if sum(i)==n])
    logpmf = [multinomial_logpmf(x[i,:], n, p) for i in range(x.shape[0])]
    return np.sum(np.exp(logpmf) * logpmf)

def multinomial_entropy_vec(n, p):
    """Vectorized version of multinomial_entropy."""
    if isinstance(n, float) or isinstance(n, int):
        return multinomial_entropy(n, p)
    else:
        n_minibatch = n.shape[0]
        return np.array([multinomial_entropy(n[i], p[i, :])
                         for i in range(n_minibatch)])

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        print(val_ed.eval())
        print(val_true)
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_entropy(n, p):
    val_true = multinomial_entropy_vec(n, p)
    _assert_eq(multinomial.entropy(n, p), val_true)
    _assert_eq(multinomial.entropy(n, tf.constant(p, dtype=tf.float32)), val_true)
    _assert_eq(multinomial.entropy(n, p), val_true)
    _assert_eq(multinomial.entropy(n, tf.constant(p, dtype=tf.float32)), val_true)

def test_entropy_1d():
    _test_entropy(1, np.array([0.5, 0.5]))
    _test_entropy(2, np.array([0.5, 0.5]))
    _test_entropy(3, np.array([0.75, 0.25]))

def test_entropy_2d():
    _test_entropy(np.array([1, 3]), np.array([[0.5, 0.5],[0.75, 0.25]]))
    _test_entropy(np.array([5, 2]), np.array([[0.5, 0.5],[0.75, 0.25]]))
