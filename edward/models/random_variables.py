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
    super(Bernoulli, self).__init__(distributions.Bernoulli, *args, **kwargs)

  @property
  def logits(self):
    return self.distribution.logits

  @property
  def p(self):
    return self.distribution.p

  @property
  def q(self):
    """1-p."""
    return self.distribution.q


class Beta(RandomVariable, distributions.Beta):
  def __init__(self, *args, **kwargs):
    super(Beta, self).__init__(distributions.Beta, *args, **kwargs)

  @property
  def a(self):
    """Shape parameter."""
    return self.distribution.a

  @property
  def b(self):
    """Shape parameter."""
    return self.distribution.b


# class Binomial(RandomVariable, distributions.Binomial):
#   def __init__(self, *args, **kwargs):
#     super(Binomial, self).__init__(distributions.Binomial, *args, **kwargs)

#   @property
#   def n(self):
#     """Number of trials."""
#     return self.distribution.n

#   @property
#   def logits(self):
#     """Log-odds."""
#     return self.distribution.logits

#   @property
#   def p(self):
#     """Probability of success."""
#     return self.distribution.p


class Categorical(RandomVariable, distributions.Categorical):
  def __init__(self, *args, **kwargs):
    super(Categorical, self).__init__(
        distributions.Categorical, *args, **kwargs)

  @property
  def num_classes(self):
    return self.distribution.num_classes

  @property
  def logits(self):
    return self.distribution.logits


class Chi2(RandomVariable, distributions.Chi2):
  def __init__(self, *args, **kwargs):
    super(Chi2, self).__init__(distributions.Chi2, *args, **kwargs)

  @property
  def df(self):
    return self.distribution.df


class Dirichlet(RandomVariable, distributions.Dirichlet):
  def __init__(self, *args, **kwargs):
    super(Dirichlet, self).__init__(distributions.Dirichlet, *args, **kwargs)

  @property
  def alpha(self):
    """Shape parameter."""
    return self.distribution.alpha


# class DirichletMultinomial(RandomVariable,
#                            distributions.DirichletMultinomial):
#   def __init__(self, *args, **kwargs):
#     super(DirichletMultinomial, self).__init__(
#         distributions.DirichletMultinomial, *args, **kwargs)

#   @property
#   def n(self):
#     """Parameter defining this distribution."""
#     return self.distribution.n

#   @property
#   def alpha(self):
#     """Parameter defining this distribution."""
#     return self.distribution.alpha


class Exponential(RandomVariable, distributions.Exponential):
  def __init__(self, *args, **kwargs):
    super(Exponential, self).__init__(
        distributions.Exponential, *args, **kwargs)

  @property
  def lam(self):
    return self.distribution.lam


class Gamma(RandomVariable, distributions.Gamma):
  def __init__(self, *args, **kwargs):
    super(Gamma, self).__init__(distributions.Gamma, *args, **kwargs)

  @property
  def alpha(self):
    """Shape parameter."""
    return self.distribution.alpha

  @property
  def beta(self):
    """Inverse scale parameter."""
    return self.distribution.beta


class InverseGamma(RandomVariable, distributions.InverseGamma):
  def __init__(self, *args, **kwargs):
    super(InverseGamma, self).__init__(
        distributions.InverseGamma, *args, **kwargs)

  @property
  def alpha(self):
    """Shape parameter."""
    return self.distribution.alpha

  @property
  def beta(self):
    """Scale parameter."""
    return self.distribution.beta


class Laplace(RandomVariable, distributions.Laplace):
  def __init__(self, *args, **kwargs):
    super(Laplace, self).__init__(distributions.Laplace, *args, **kwargs)

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self.distribution.scale


class Mixture(RandomVariable, distributions.Mixture):
  def __init__(self, *args, **kwargs):
    super(Mixture, self).__init__(
        distributions.Mixture, *args, **kwargs)

  @property
  def cat(self):
    return self.mixture.cat

  @property
  def components(self):
    return self.mixture.components

  @property
  def num_components(self):
    return self.mixture.num_components


# class Multinomial(RandomVariable, distributions.Multinomial):
#   def __init__(self, *args, **kwargs):
#     super(Multinomial, self).__init__(
#         distributions.Multinomial, *args, **kwargs)

#   @property
#   def n(self):
#     """Number of trials."""
#     return self.distribution.n

#   @property
#   def p(self):
#     """Event probabilities."""
#     return self.distribution.p

#   @property
#   def logits(self):
#     """Log-odds."""
#     return self.distribution.logits


class MultivariateNormalCholesky(RandomVariable,
                                 distributions.MultivariateNormalCholesky):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalCholesky, self).__init__(
        distributions.MultivariateNormalCholesky, *args, **kwargs)

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class MultivariateNormalDiag(RandomVariable,
                             distributions.MultivariateNormalDiag):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalDiag, self).__init__(
        distributions.MultivariateNormalDiag, *args, **kwargs)

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class MultivariateNormalDiagPlusVDVT(RandomVariable,
                                     distributions.
                                     MultivariateNormalDiagPlusVDVT):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalDiagPlusVDVT, self).__init__(
        distributions.MultivariateNormalDiagPlusVDVT, *args, **kwargs)

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class MultivariateNormalFull(RandomVariable,
                             distributions.MultivariateNormalFull):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalFull, self).__init__(
        distributions.MultivariateNormalFull, *args, **kwargs)

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class Normal(RandomVariable, distributions.Normal):
  def __init__(self, *args, **kwargs):
    super(Normal, self).__init__(distributions.Normal, *args, **kwargs)

  @property
  def mu(self):
    """Distribution parameter for the mean."""
    return self.distribution.mu

  @property
  def sigma(self):
    """Distribution parameter for standard deviation."""
    return self.distribution.sigma


# class Poisson(RandomVariable, distributions.Poisson):
#   def __init__(self, *args, **kwargs):
#     super(Poisson, self).__init__(distributions.Poisson, *args, **kwargs)

#   @property
#   def lam(self):
#     """Rate parameter."""
#     return self.distribution.lam


class QuantizedDistribution(RandomVariable,
                            distributions.QuantizedDistribution):
  def __init__(self, *args, **kwargs):
    super(QuantizedDistribution, self).__init__(
        distributions.QuantizedDistribution, *args, **kwargs)

  @property
  def base_distribution(self):
    """Base distribution, p(x)."""
    return self.distribution.base_distribution


class StudentT(RandomVariable, distributions.StudentT):
  def __init__(self, *args, **kwargs):
    super(StudentT, self).__init__(distributions.StudentT, *args, **kwargs)

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self.distribution.df

  @property
  def mu(self):
    """Locations of these Student's t distribution(s)."""
    return self.distribution.mu

  @property
  def sigma(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self.distribution.sigma


class TransformedDistribution(RandomVariable,
                              distributions.TransformedDistribution):
  def __init__(self, *args, **kwargs):
    super(TransformedDistribution, self).__init__(
        distributions.TransformedDistribution, *args, **kwargs)

  @property
  def base_distribution(self):
    """Base distribution, p(x)."""
    return self.distribution.base_distribution

  @property
  def transform(self):
    """Function transforming x => y."""
    return self.distribution.transform

  @property
  def inverse(self):
    """Inverse function of transform, y => x."""
    return self.distribution.inverse

  @property
  def log_det_jacobian(self):
    """Function computing the log determinant of the Jacobian of transform."""
    return self.distribution.log_det_jacobian


class Uniform(RandomVariable, distributions.Uniform):
  def __init__(self, *args, **kwargs):
    super(Uniform, self).__init__(distributions.Uniform, *args, **kwargs)

  @property
  def a(self):
    return self.distribution.a

  @property
  def b(self):
    return self.distribution.b


class WishartCholesky(RandomVariable, distributions.WishartCholesky):
  def __init__(self, *args, **kwargs):
    super(WishartCholesky, self).__init__(
        distributions.WishartCholesky, *args, **kwargs)

  @property
  def df(self):
    """Wishart distribution degree(s) of freedom."""
    return self.distribution.df

  def scale(self):
    """Wishart distribution scale matrix."""
    return self.distribution.scale()

  @property
  def scale_operator_pd(self):
    """Wishart distribution scale matrix as an OperatorPD."""
    return self.distribution.scale_operator_pd

  @property
  def cholesky_input_output_matrices(self):
    """Boolean indicating if `Tensor` input/outputs are Cholesky factorized."""
    return self.distribution.cholesky_input_output_matrices

  @property
  def dimension(self):
    """Dimension of underlying vector space. The `p` in `R^(p*p)`."""
    return self.distribution.dimension


class WishartFull(RandomVariable, distributions.WishartFull):
  def __init__(self, *args, **kwargs):
    super(WishartFull, self).__init__(
        distributions.WishartFull, *args, **kwargs)

  @property
  def df(self):
    """Wishart distribution degree(s) of freedom."""
    return self.distribution.df

  def scale(self):
    """Wishart distribution scale matrix."""
    return self.distribution.scale()

  @property
  def scale_operator_pd(self):
    """Wishart distribution scale matrix as an OperatorPD."""
    return self.distribution.scale_operator_pd

  @property
  def cholesky_input_output_matrices(self):
    """Boolean indicating if `Tensor` input/outputs are Cholesky factorized."""
    return self.distribution.cholesky_input_output_matrices

  @property
  def dimension(self):
    """Dimension of underlying vector space. The `p` in `R^(p*p)`."""
    return self.distribution.dimension


class Empirical(RandomVariable, distributions_Empirical):
  def __init__(self, *args, **kwargs):
    super(Empirical, self).__init__(distributions_Empirical, *args, **kwargs)

  @property
  def params(self):
    """Distribution parameter."""
    return self.distribution.params


class PointMass(RandomVariable, distributions_PointMass):
  def __init__(self, *args, **kwargs):
    super(PointMass, self).__init__(distributions_PointMass, *args, **kwargs)

  @property
  def params(self):
    """Distribution parameter."""
    return self.distribution.params
