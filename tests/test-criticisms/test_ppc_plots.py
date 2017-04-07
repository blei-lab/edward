from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg', warn=False)

import numpy as np

from edward.criticisms.ppc_plots import ppc_density_plot, ppc_stat_hist_plot


def test_ppc_density_plot():
  y = np.random.randn(20)
  y_rep = np.random.randn(20, 20)

  ppc_density_plot(y, y_rep)


def test_ppc_stat_hist_plot():
  y = np.random.randn(20)
  t = 0.0

  ppc_stat_hist_plot(t, y, stat_name="mean", bins=10)
