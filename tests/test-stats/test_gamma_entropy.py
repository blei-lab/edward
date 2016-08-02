from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import gamma
from scipy import stats

def gamma_entropy_vec(a, scale):
    """Vectorized version of stats.gamma.entropy."""
    if isinstance(scale, float):
        return stats.gamma.entropy(a, scale=scale)
    else:
        return np.array([stats.gamma.entropy(a_x, scale=scale_x)
                         for a_x, scale_x in zip(a, scale)])


class test_gamma_entropy_class(tf.test.TestCase):

    def _test(self, a, scale=1):
        val_true = gamma_entropy_vec(a, scale=scale)
        with self.test_session():
            self.assertAllClose(gamma.entropy(a, scale).eval(), val_true)
            self.assertAllClose(gamma.entropy(tf.constant(a), tf.constant(scale)).eval(), val_true)


    def test_0d(self):
        self._test(a=1.0, scale=1.0)
        self._test(a=1.0, scale=1.0)

        self._test(a=0.5, scale=5.0)
        self._test(a=5.0, scale=0.5)


    def test_1d(self):
        self._test(a=[0.5, 1.2, 5.3, 8.7], scale=[0.5, 1.2, 5.3, 8.7])
