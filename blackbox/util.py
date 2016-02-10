import math
import numpy as np

def logistic(x):
  vals = x < -300
  if (x < -300):
    return 1e-300;
  return 1.0 / (1.0 + math.exp(-x))

def logistic_deriv(x):
  return logistic(x) * (1 - logistic(x))

def logistic_hess(x):
  return logistic_deriv(x) * (1 - 2 * logistic(x))

def tanh(x):
  return 2 * logistic(2 * x) -1

def tanh_deriv(x):
  return 4 * logistic_deriv(2 * x)

def tanh_hess(x):
  return 8 * logistic_hess(2 * x)

def h(x):
  return tanh(x)

def h_deriv(x):
  return tanh_deriv(x)

def h_hess(x):
  return tanh_hess(x)

def discrete_density(samples, support=None):
  """
  Construct discrete pmf p, with output as a supp(p_1) x ... x
  supp(p_d) probability table.

  Arguments
  ----------
  samples: N x d matrix
  support: list of the form [[a1, b1], ..., [ad, bd]].
           Force support to be in these intervals, and drop any
           sample outside the support. Default is
           supp(p_i) = [min(p_i), min(p_i)+1, min(p_i)+2, ..., max(p_i)]
           for each dimension.
  """
  N = samples.shape[0]
  D = samples.shape[1]
  bins = [0]*D
  for d in range(D):
    if support is None:
      bin_edges = np.arange(int(np.min(samples[:, d])),
                            int(np.max(samples[:, d]))+1)
    else:
      bin_edges = np.arange(support[d][0], support[d][1]+1)

    bin_edges = bin_edges + 0.5
    bin_edges[-1] = bin_edges[-1] - 0.5
    bin_edges = np.insert(bin_edges, 0, 0)
    bins[d] = list(bin_edges)

  density = np.histogramdd(samples, bins=tuple(bins))[0]
  return density / N
