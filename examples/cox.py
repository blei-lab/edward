#!/usr/bin/env python
"""Cox process model using mean-field variational inference.
   Ref: Miller et al, 2014.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import poisson, multivariate_normal
from edward.util import multivariate_rbf
from scipy.stats import multivariate_normal as mvn, poisson as poi

class CoxProcess:
  """
  Cox process model
  x = (x_1, ..., x_N)...each x_i is a set of counts
  z = (z_1, ..., z_N)...each z_i is the intensity function for x_i

  p(x, z) = \prod_{i=1}^{N} p(x_i | z_i) p(z_i)
  	
  p(z_i) = N(z_i | 0, K)
  p(x_i | z_i) = \prod_{v=1}^{V} p(x_{i,v} | z_{i,v})
  p(x_{i,v} | z_{i,v}) = Poisson(x_{i,v} | exp(z_{i,v} + f_0)) 

  K is the kernel of the GP. Its parameters are assumed to be known.

  Parameters
  ----------
  N : int
    Number of data points.
  V : int
  	Size of the set counts for each data point.
  sigma : float, optional
    Signal variance parameter.
  l : float, optional
    Length scale parameter.
  """
  def __init__(self, N, V, sigma=1.0, l=1.0):
    self.N = N
    self.V = V
    self.sigma = sigma
    self.l = l

    self.n_vars = N * V
    self.inverse_link = tf.exp

  def kernel_row(self, x):
  	"""
  	computes the covariance matrix using rbf for the row vector x
  	Args:
  	x: a row vector fom the data
  	   size: V
  	Returns:
  	   mat: a matrix pf size VxV
  	"""
  	mat = []
	for i in range(self.V):
	  vect = []
	  xi = x[i]
	  for j in range(self.V):
	    if j == i:
	      vect.append(multivariate_rbf(xi, xi, self.sigma, self.l))
	    else:
	      xj = x[j]
	      vect.append(multivariate_rbf(xi, xj, self.sigma, self.l))

	  mat.append(vect)

	mat = tf.pack(mat)

	return mat

  def kernel(self, x):
  	"""
  	wrapper around kernel_row that computes K for x
  	Args:
  	x: the whole data
  	   size: NxV
  	Returns:
  	K: the kernel matrix for the whole data
  	   size: NxVxV
  	"""
  	K = [] 	
	for k in range(self.N):
		xk = x[k, :]
		K.append(self.kernel_row(xk))

	K = tf.pack(K)

	return K

  def log_prob(self, xs, zs):
    """
    Args:
    xs: the data
    	size NxV
    zs: latent variables
    	size NxV
    Returns: log joint log p(xs, zs), scalar	

    TODO: take care of cases where K is ill-conditioned
    		for now hack on toy data is to add a small constant
    """
    x = xs['x']
    z = zs['z']
    z = tf.reshape(z, [self.N, self.V])

    log_prior = multivariate_normal.logpdf(
			z, tf.zeros([self.N, self.V]), 1e-2+self.kernel(x))

    log_lik = poisson.logpmf(x, mu=self.inverse_link(z))
    log_lik = tf.reduce_sum(log_lik, 1)

    return tf.reduce_sum(log_prior + log_lik)
    
def build_toy_dataset(N, V):
	"""
	toy data...Identity kernel for GP
	"""
	K = np.identity(V)
	x = np.zeros([N, V])
	for i in range(N):
		z_i = mvn.rvs(cov=K, size=1)
		for v in range(V):
			lam = np.exp(z_i[v])
			x[i, v] = poi.rvs(mu=lam, size=1)

	print("toy data: {}".format(x))

	return x

ed.set_seed(42)
sess = ed.get_session()
df = build_toy_dataset(N=5, V=2)
data = {'x': df}
N = len(df)
V = len(df[0, :])
model = CoxProcess(N=N, V=V)

qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

inference = ed.MFVI({'z': qz}, data, model)
inference.initialize()

init = tf.initialize_all_variables()
init.run()

n_iter = 5000
for t in range(inference.n_iter):
  info_dict = inference.update()
  if t % inference.n_print == 0:
  	mean, std = sess.run([qz.mu, qz.sigma])
  	print("iter: {}...loss: {}...mean: {}...std: {}".format(
  			info_dict['t'], info_dict['loss'], mean, std))
