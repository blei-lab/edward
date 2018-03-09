from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal, Poisson


class test_call_with_manipulate_class(tf.test.TestCase):

  def _test_intercept_value(self, RV, value, *args, **kwargs):
    def manipulate(f, *fargs, **fkwargs):
      name = kwargs.get('name', None)
      if name == "rv2":
        kwargs['value'] = rv1.value
      return f(*fargs, **fkwargs)
    rv1 = RV(*args, value=value, name="rv1", **kwargs)
    rv2 = ed.call_with_manipulate(RV, manipulate, *args, name="rv2", **kwargs)
    value_shape1 = rv1.value.shape
    value_shape2 = rv2.value.shape
    self.assertEqual(value_shape1, value_shape2)

  def test_intercept_value(self):
    with self.test_session():
      self._test_intercept_value(Normal, 2, loc=0.5, scale=1.0)
      self._test_intercept_value(Normal, [2], loc=[0.5], scale=[1.0])
      self._test_intercept_value(Poisson, 2, rate=0.5)

if __name__ == '__main__':
  tf.test.main()
