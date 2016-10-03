from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.empirical import Empirical as distributions_Empirical
from edward.models.point_mass import PointMass as distributions_PointMass
from edward.models.random_variable import RandomVariable
from edward.util import get_session

distributions = tf.contrib.distributions


class Bernoulli(RandomVariable, distributions.Bernoulli):
  def __init__(self, *args, **kwargs):
    super(Bernoulli, self).__init__(*args, **kwargs)


class Beta(RandomVariable, distributions.Beta):
  def __init__(self, *args, **kwargs):
    super(Beta, self).__init__(*args, **kwargs)


# class Binomial(RandomVariable, distributions.Binomial):
#   def __init__(self, *args, **kwargs):
#     super(Binomial, self).__init__(*args, **kwargs)


class Categorical(RandomVariable, distributions.Categorical):
  def __init__(self, *args, **kwargs):
    super(Categorical, self).__init__(*args, **kwargs)


class Chi2(RandomVariable, distributions.Chi2):
  def __init__(self, *args, **kwargs):
    super(Chi2, self).__init__(*args, **kwargs)


class Dirichlet(RandomVariable, distributions.Dirichlet):
  def __init__(self, *args, **kwargs):
    super(Dirichlet, self).__init__(*args, **kwargs)


# class DirichletMultinomial(RandomVariable,
#                            distributions.DirichletMultinomial):
#   def __init__(self, *args, **kwargs):
#     super(DirichletMultinomial, self).__init__(*args, **kwargs)


class Exponential(RandomVariable, distributions.Exponential):
  def __init__(self, *args, **kwargs):
    super(Exponential, self).__init__(*args, **kwargs)


class Gamma(RandomVariable, distributions.Gamma):
  def __init__(self, *args, **kwargs):
    super(Gamma, self).__init__(*args, **kwargs)


class InverseGamma(RandomVariable, distributions.InverseGamma):
  def __init__(self, *args, **kwargs):
    super(InverseGamma, self).__init__(*args, **kwargs)


class Laplace(RandomVariable, distributions.Laplace):
  def __init__(self, *args, **kwargs):
    super(Laplace, self).__init__(*args, **kwargs)


class Mixture(RandomVariable, distributions.Mixture):
  def __init__(self, *args, **kwargs):
    super(Mixture, self).__init__(*args, **kwargs)


# class Multinomial(RandomVariable, distributions.Multinomial):
#   def __init__(self, *args, **kwargs):
#     super(Multinomial, self).__init__(*args, **kwargs)


class MultivariateNormalCholesky(RandomVariable,
                                 distributions.MultivariateNormalCholesky):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalCholesky, self).__init__(*args, **kwargs)


class MultivariateNormalDiag(RandomVariable,
                             distributions.MultivariateNormalDiag):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalDiag, self).__init__(*args, **kwargs)


class MultivariateNormalDiagPlusVDVT(RandomVariable,
                                     distributions.
                                     MultivariateNormalDiagPlusVDVT):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalDiagPlusVDVT, self).__init__(*args, **kwargs)


class MultivariateNormalFull(RandomVariable,
                             distributions.MultivariateNormalFull):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalFull, self).__init__(*args, **kwargs)


class Normal(RandomVariable, distributions.Normal):
  def __init__(self, *args, **kwargs):
    super(Normal, self).__init__(*args, **kwargs)


# class Poisson(RandomVariable, distributions.Poisson):
#   def __init__(self, *args, **kwargs):
#     super(Poisson, self).__init__(*args, **kwargs)


class QuantizedDistribution(RandomVariable,
                            distributions.QuantizedDistribution):
  def __init__(self, *args, **kwargs):
    super(QuantizedDistribution, self).__init__(*args, **kwargs)


class StudentT(RandomVariable, distributions.StudentT):
  def __init__(self, *args, **kwargs):
    super(StudentT, self).__init__(*args, **kwargs)


class TransformedDistribution(RandomVariable,
                              distributions.TransformedDistribution):
  def __init__(self, *args, **kwargs):
    super(TransformedDistribution, self).__init__(*args, **kwargs)


class Uniform(RandomVariable, distributions.Uniform):
  def __init__(self, *args, **kwargs):
    super(Uniform, self).__init__(*args, **kwargs)


class WishartCholesky(RandomVariable, distributions.WishartCholesky):
  def __init__(self, *args, **kwargs):
    super(WishartCholesky, self).__init__(*args, **kwargs)


class WishartFull(RandomVariable, distributions.WishartFull):
  def __init__(self, *args, **kwargs):
    super(WishartFull, self).__init__(*args, **kwargs)


class Empirical(RandomVariable, distributions_Empirical):
  def __init__(self, *args, **kwargs):
    super(Empirical, self).__init__(*args, **kwargs)


class PointMass(RandomVariable, distributions_PointMass):
  def __init__(self, *args, **kwargs):
    super(PointMass, self).__init__(*args, **kwargs)
