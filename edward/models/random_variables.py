from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.empirical import Empirical as distributions_Empirical
from edward.models.point_mass import PointMass as distributions_PointMass
from edward.models.random_variable import RandomVariable
from edward.util import get_session

distributions = tf.contrib.distributions


class Bernoulli(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Bernoulli, self).__init__(distributions.Bernoulli, *args, **kwargs)

  def __str__(self):
    try:
      p = self.p.eval()
      return "p: \n" + p.__str__()
    except:
      return super(Bernoulli, self).__str__()

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


class Beta(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Beta, self).__init__(distributions.Beta, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      a, b = sess.run([self.a, self.b])
      return "a: \n" + a.__str__() + "\n" + \
             "b: \n" + b.__str__()
    except:
      return super(Beta, self).__str__()

  @property
  def a(self):
    """Shape parameter."""
    return self.distribution.a

  @property
  def b(self):
    """Shape parameter."""
    return self.distribution.b


class Categorical(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Categorical, self).__init__(
        distributions.Categorical, *args, **kwargs)

  def __str__(self):
    try:
      logits = self.logits.eval()
      return "logits: \n" + logits.__str__()
    except:
      return super(Categorical, self).__str__()

  @property
  def num_classes(self):
    return self.distribution.num_classes

  @property
  def logits(self):
    return self.distribution.logits


class Chi2(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Chi2, self).__init__(distributions.Chi2, *args, **kwargs)

  def __str__(self):
    try:
      df = self.df.eval()
      return "df: \n" + df.__str__()
    except:
      return super(Chi2, self).__str__()

  @property
  def df(self):
    return self.distribution.df


class Dirichlet(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Dirichlet, self).__init__(distributions.Dirichlet, *args, **kwargs)

  def __str__(self):
    try:
      alpha = self.alpha.eval()
      return "alpha: \n" + alpha.__str__()
    except:
      return super(Dirichlet, self).__str__()

  @property
  def alpha(self):
    """Shape parameter."""
    return self.distribution.alpha


class DirichletMultinomial(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(DirichletMultinomial, self).__init__(
        distributions.DirichletMultinomial, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      n, alpha = sess.run([self.n, self.alpha])
      return "n: \n" + n.__str__() + "\n" + \
             "alpha: \n" + alpha.__str__()
    except:
      return super(DirichletMultinomial, self).__str__()

  @property
  def n(self):
    """Parameter defining this distribution."""
    return self.distribution.n

  @property
  def alpha(self):
    """Parameter defining this distribution."""
    return self.distribution.alpha


class Exponential(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Exponential, self).__init__(
        distributions.Exponential, *args, **kwargs)

  def __str__(self):
    try:
      lam = self.lam.eval()
      return "lam: \n" + lam.__str__()
    except:
      return super(Exponential, self).__str__()

  @property
  def lam(self):
    return self.distribution.lam


class Gamma(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Gamma, self).__init__(distributions.Gamma, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      alpha, beta = sess.run([self.alpha, self.beta])
      return "alpha: \n" + alpha.__str__() + "\n" + \
             "beta: \n" + beta.__str__()
    except:
      return super(Gamma, self).__str__()

  @property
  def alpha(self):
    """Shape parameter."""
    return self.distribution.alpha

  @property
  def beta(self):
    """Inverse scale parameter."""
    return self.distribution.beta


class InverseGamma(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(InverseGamma, self).__init__(
        distributions.InverseGamma, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      alpha, beta = sess.run([self.alpha, self.beta])
      return "alpha: \n" + alpha.__str__() + "\n" + \
             "beta: \n" + beta.__str__()
    except:
      return super(InverseGamma, self).__str__()

  @property
  def alpha(self):
    """Shape parameter."""
    return self.distribution.alpha

  @property
  def beta(self):
    """Scale parameter."""
    return self.distribution.beta


class Laplace(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Laplace, self).__init__(distributions.Laplace, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      loc, scale = sess.run([self.loc, self.scale])
      return "loc: \n" + loc.__str__() + "\n" + \
             "scale: \n" + scale.__str__()
    except:
      return super(Laplace, self).__str__()

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self.distribution.scale


class MultivariateNormalCholesky(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalCholesky, self).__init__(
        distributions.MultivariateNormalCholesky, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      mu, sigma = sess.run([self.mu, self.sigma])
      return "mu: \n" + mu.__str__() + "\n" + \
             "sigma: \n" + sigma.__str__()
    except:
      return super(MultivariateNormalCholesky, self).__str__()

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class MultivariateNormalDiag(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalDiag, self).__init__(
        distributions.MultivariateNormalDiag, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      mu, sigma = sess.run([self.mu, self.sigma])
      return "mu: \n" + mu.__str__() + "\n" + \
             "sigma: \n" + sigma.__str__()
    except:
      return super(MultivariateNormalDiag, self).__str__()

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class MultivariateNormalFull(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(MultivariateNormalFull, self).__init__(
        distributions.MultivariateNormalFull, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      mu, sigma = sess.run([self.mu, self.sigma])
      return "mu: \n" + mu.__str__() + "\n" + \
             "sigma: \n" + sigma.__str__()
    except:
      return super(MultivariateNormalFull, self).__str__()

  @property
  def mu(self):
    return self.distribution.mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    return self.distribution.sigma


class Normal(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Normal, self).__init__(distributions.Normal, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      mu, sigma = sess.run([self.mu, self.sigma])
      return "mu: \n" + mu.__str__() + "\n" + \
             "sigma: \n" + sigma.__str__()
    except:
      return super(Normal, self).__str__()

  @property
  def mu(self):
    """Distribution parameter for the mean."""
    return self.distribution.mu

  @property
  def sigma(self):
    """Distribution parameter for standard deviation."""
    return self.distribution.sigma


class StudentT(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(StudentT, self).__init__(distributions.StudentT, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      df, mu, sigma = sess.run([self.df, self.mu, self.sigma])
      return "df: \n" + df.__str__() + "\n" + \
             "mu: \n" + mu.__str__() + "\n" + \
             "sigma: \n" + sigma.__str__()
    except:
      return super(StudentT, self).__str__()

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


class TransformedDistribution(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(TransformedDistribution, self).__init__(
        distributions.TransformedDistribution, *args, **kwargs)

  def __str__(self):
    try:
      return self.base_distribution.__str__()
    except:
      return super(TransformedDistribution, self).__str__()

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


class Uniform(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Uniform, self).__init__(distributions.Uniform, *args, **kwargs)

  def __str__(self):
    try:
      sess = get_session()
      a, b = sess.run([self.a, self.b])
      return "a: \n" + a.__str__() + "\n" + \
             "b: \n" + b.__str__()
    except:
      return super(Uniform, self).__str__()

  @property
  def a(self):
    return self.distribution.a

  @property
  def b(self):
    return self.distribution.b


class Empirical(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(Empirical, self).__init__(distributions_Empirical, *args, **kwargs)

  def __str__(self):
    try:
      mean = self.mean().eval()
      return "mean: \n" + mean.__str__()
    except:
      return super(Empirical, self).__str__()

  @property
  def params(self):
    """Distribution parameter."""
    return self.distribution.params


class PointMass(RandomVariable):
  def __init__(self, *args, **kwargs):
    super(PointMass, self).__init__(distributions_PointMass, *args, **kwargs)

  def __str__(self):
    try:
      params = self.params.eval()
      return "params: \n" + params.__str__()
    except:
      return super(PointMass, self).__str__()

  @property
  def params(self):
    """Distribution parameter."""
    return self.distribution.params
