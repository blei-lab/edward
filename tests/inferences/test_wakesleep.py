from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal


class test_wakesleep_class(tf.test.TestCase):

  def _test_model_parameter(self, Inference, *args, **kwargs):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

      p = tf.sigmoid(tf.Variable(0.5))
      x = Bernoulli(probs=p, sample_shape=10)

      inference = Inference({}, data={x: x_data})
      inference.run(*args, **kwargs)

      self.assertAllClose(p.eval(), 0.2, rtol=5e-2, atol=5e-2)

  def test_wakesleep(self):
    self._test_model_parameter(ed.WakeSleep, n_iter=50)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
