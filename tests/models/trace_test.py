from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal, Poisson


class test_trace_class(tf.test.TestCase):

  def _test_intercept_value(self, RV, value, *args, **kwargs):
    def _intercept(f, *args, **kwargs):
      name = kwargs.get('name', None)
      if name == "rv2":
        rv1 = rv1_trace["rv1"].value
        kwargs['value'] = rv1.value
      return f(*args, **kwargs)
    with ed.Trace() as rv1_trace:
      rv1 = RV(*args, value=value, name="rv1", **kwargs)
    with ed.Trace(intercept=_intercept) as rv2_trace:
      rv2 = RV(*args, name="rv2", **kwargs)
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
