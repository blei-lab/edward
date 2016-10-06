from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.util import get_dims, tile


class test_tile_class(tf.test.TestCase):

  def _test(self, input, multiples):
    if isinstance(multiples, int) or isinstance(multiples, float):
      multiples_shape = [multiples]
    elif isinstance(multiples, tuple):
      multiples_shape = list(multiples)
    else:
      multiples_shape = multiples

    input_shape = get_dims(input)
    diff = len(input_shape) - len(multiples_shape)
    if diff < 0:
      input_shape = [1] * abs(diff) + input_shape
    elif diff > 0:
      multiples_shape = [1] * diff + multiples_shape

    val_true = [x * y for x, y in zip(input_shape, multiples_shape)]
    with self.test_session():
      val_est = get_dims(tile(input, multiples))
      assert val_est == val_true

  def test_0d(self):
    x = tf.constant(0)
    self._test(x, 2)
    self._test(x, (2, 1))

  def test_1d(self):
    x = tf.constant([0, 1, 2])
    self._test(x, 2)
    self._test(x, (2, 2))
    self._test(x, (2, 1, 2))
    x = tf.constant([1, 2, 3, 4])
    self._test(x, (4, 1))

  def test_2d(self):
    x = tf.constant([[1, 2], [3, 4]])
    self._test(x, 2)
    self._test(x, (2, 1))

if __name__ == '__main__':
  tf.test.main()
