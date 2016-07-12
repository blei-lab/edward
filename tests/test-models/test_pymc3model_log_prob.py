from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import pymc3 as pm
import six
import tensorflow as tf
import theano

from edward.models import PyMC3Model

sess = tf.Session()


def _test(model, data, zs):
    val_ed = model.log_prob(data, zs)
    model.keys = list(six.iterkeys(data))
    model.values = list(six.itervalues(data))
    val_true = model._py_log_prob_args(zs)
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)
        zs_tf = tf.constant(zs, dtype=tf.float32)
        val_ed = model.log_prob(data, zs_tf)
        assert np.allclose(val_ed.eval(), val_true)


def test_1d():
    x_obs = theano.shared(np.zeros(1))
    with pm.Model() as pm_model:
        beta = pm.Beta('beta', 1, 1, transform=None)
        x = pm.Bernoulli('x', beta, observed=x_obs)

    model = PyMC3Model(pm_model)
    data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
    zs = np.array([[0.5]])
    _test(model, data, zs)
    zs = np.array([[0.4], [0.2], [0.2351], [0.6213]])
    _test(model, data, zs)
