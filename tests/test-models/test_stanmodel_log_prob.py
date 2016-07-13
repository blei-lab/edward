from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import pystan
import tensorflow as tf

sess = tf.Session()


def log_prob(zs, stanfit):
    lp = np.zeros(zs.shape[0], dtype=np.float32)
    for b, z in enumerate(zs):
        z_unconst = stanfit.unconstrain_pars({'theta': z[0]})
        lp[b] = stanfit.log_prob(z_unconst, adjust_transform=False)

    return lp


def _test(ed_model, pystan_model, data, zs):
    stanfit = pystan_model.sampling(data=data, iter=1, chains=1)
    val_ed = ed_model.log_prob(data, zs)
    val_true = log_prob(zs, stanfit)
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)
        zs_tf = tf.constant(zs, dtype=tf.float32)
        val_ed = ed_model.log_prob(data, zs_tf)
        assert np.allclose(val_ed.eval(), val_true)


def test_1d():
    model_code = """
        data {
          int<lower=0> N;
          int<lower=0,upper=1> y[N];
        }
        parameters {
          real<lower=0,upper=1> theta;
        }
        model {
          theta ~ beta(0.5, 0.5);  // Jeffreys' prior
          for (n in 1:N)
            y[n] ~ bernoulli(theta);
        }
    """
    pystan_model = pystan.StanModel(model_code=model_code)
    ed_model = ed.StanModel(model=pystan_model)
    data = {'N': 10, 'y': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]}
    zs = np.array([[0.5]])
    _test(ed_model, pystan_model, data, zs)
    zs = np.array([[0.4], [0.2], [0.2351], [0.6213]])
    _test(ed_model, pystan_model, data, zs)
