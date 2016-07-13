from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import PythonModel
from scipy.stats import beta, bernoulli

sess = tf.Session()


class BetaBernoulli(PythonModel):
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def _py_log_prob(self, xs, zs):
        # This example is written for pedagogy. We recommend
        # vectorizing operations in practice.
        n_minibatch = zs.shape[0]
        lp = np.zeros(n_minibatch, dtype=np.float32)
        for b in range(n_minibatch):
            lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
            for n in range(xs['x'].shape[0]):
                lp[b] += bernoulli.logpmf(xs['x'][n], p=zs[b, :])

        return lp


def _test(model, data, zs):
    val_ed = model.log_prob(data, zs)
    val_true = model._py_log_prob(data, zs)
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)
        zs_tf = tf.constant(zs, dtype=tf.float32)
        val_ed = model.log_prob(data, zs_tf)
        assert np.allclose(val_ed.eval(), val_true)


def test_1d():
    model = BetaBernoulli()
    data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
    zs = np.array([[0.5]])
    _test(model, data, zs)
    zs = np.array([[0.4], [0.2], [0.2351], [0.6213]])
    _test(model, data, zs)
