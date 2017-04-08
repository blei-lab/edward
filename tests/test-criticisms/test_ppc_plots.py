from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.criticisms.ppc_plots import ppc_density_plot, ppc_stat_hist_plot


class test_ppc_plots_class(tf.test.TestCase):
  def test_ppc_density_plot(self):
    y = np.random.randn(20)
    y_rep = np.random.randn(20, 20)

    ppc_density_plot(y, y_rep)

  def test_ppc_stat_hist_plot(self):
    y = np.random.randn(20)
    t = 0.0

    ppc_stat_hist_plot(t, y, stat_name="mean", bins=10)

if __name__ == '__main__':
    tf.test.main()
