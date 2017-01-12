from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli as tfBernoulli

from edward.models import Bernoulli


class test_bernoulli_doc_class(tf.test.TestCase):

  def test_0d(self):
    assert len(Bernoulli.__doc__) > 0
    assert Bernoulli.__doc__ == tfBernoulli.__doc__
    assert Bernoulli.__name__ == "Bernoulli"

if __name__ == '__main__':
  tf.test.main()
