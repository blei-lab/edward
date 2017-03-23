from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli
from tensorflow.contrib import distributions as ds


class test_bernoulli_doc_class(tf.test.TestCase):

  def test(self):
    self.assertGreater(len(Bernoulli.__doc__), 0)
    self.assertEqual(Bernoulli.__doc__, ds.Bernoulli.__doc__)
    self.assertEqual(Bernoulli.__name__, "Bernoulli")

if __name__ == '__main__':
  tf.test.main()
