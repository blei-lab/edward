from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


class Bernoulli(sg.DistributionTensor):
  """
  Examples
  --------
  >>> p = tf.constant([0.5])
  >>> x = Bernoulli(p=p)
  >>>
  >>> z1 = tf.constant([[2.0, 8.0]])
  >>> z2 = tf.constant([[1.0, 2.0]])
  >>> x = Bernoulli(p=tf.matmul(z1, z2))
  """
  def __init__(self, *args, **kwargs):
    super(Bernoulli, self).__init__(distributions.Bernoulli, *args, **kwargs)


class Beta(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Beta, self).__init__(distributions.Beta, *args, **kwargs)


class Categorical(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Categorical, self).__init__(distributions.Categorical, *args, **kwargs)


class Chi2(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Chi2, self).__init__(distributions.Chi2, *args, **kwargs)


class Dirichlet(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Dirichlet, self).__init__(distributions.Dirichlet, *args, **kwargs)


class DirichletMultinomial(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(DirichletMultinomial, self).__init__(distributions.DirichletMultinomial, *args, **kwargs)


class Exponential(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Exponential, self).__init__(distributions.Exponential, *args, **kwargs)


class Gamma(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Gamma, self).__init__(distributions.Gamma, *args, **kwargs)


class InverseGamma(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(InverseGamma, self).__init__(distributions.InverseGamma, *args, **kwargs)


class Laplace(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Laplace, self).__init__(distributions.Laplace, *args, **kwargs)


class MultivariateNormalCholesky(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalCholesky, self).__init__(distributions.MultivariateNormalCholesky, *args, **kwargs)


class MultivariateNormalDiag(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalDiag, self).__init__(distributions.MultivariateNormalDiag, *args, **kwargs)


class MultivariateNormalFull(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalFull, self).__init__(distributions.MultivariateNormalFull, *args, **kwargs)


class Normal(sg.DistributionTensor):
  """
  Examples
  --------
  >>> mu = Normal(mu=tf.constant(0.0), sigma=tf.constant(1.0)])
  >>> x = Normal(mu=mu, sigma=tf.constant([1.0]))
  """
  def __init__(self, *args, **kwargs):
    super(Normal, self).__init__(distributions.Normal, *args, **kwargs)


class StudentT(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(StudentT, self).__init__(distributions.StudentT, *args, **kwargs)


class TransformedDistribution(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(TransformedDistribution, self).__init__(distributions.TransformedDistribution, *args, **kwargs)


class Uniform(sg.DistributionTensor):
  def __init__(self, *args, **kwargs):
    super(Uniform, self).__init__(distributions.Uniform, *args, **kwargs)
