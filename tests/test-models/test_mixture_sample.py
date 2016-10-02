from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal
from edward.util import get_dims


def _test(cat, components, n):
  x = Mixture(cat=cat, components=components)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(components[0].mu)
  assert val_est == val_true


class test_mixture_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      cat = Categorical(logits=tf.zeros(2))
      components = [Normal(mu=0.0, sigma=1.0),
                    Normal(mu=0.0, sigma=1.0)]
      _test(cat, components, [1])
      _test(cat, components, [5])

  # def test_1d(self):
  #   with self.test_session():
  #     cat = Categorical(logits=tf.zeros(2))
  #     components = [Normal(mu=tf.zeros(5), sigma=tf.ones(5)),
  #                   Normal(mu=tf.zeros(5), sigma=tf.ones(5))]
  #     _test(cat, components, [1])
  #     _test(cat, components, [5])

if __name__ == '__main__':
  tf.test.main()
