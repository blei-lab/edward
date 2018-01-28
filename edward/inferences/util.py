from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import six
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models.core import TransformedDistribution

tfb = tf.contrib.distributions.bijectors


def call_function_up_to_args(f, *args, **kwargs):
  """Call f, removing any args/kwargs it doesn't take as input."""
  if hasattr(f, "_func"):  # tf.make_template()
    argspec = inspect.getargspec(f._func)
  else:
    argspec = inspect.getargspec(f)
  fkwargs = {}
  for k, v in six.iteritems(kwargs):
    if k in argspec.args:
      fkwargs[k] = v
  num_args = len(argspec.args) - len(fkwargs)
  if num_args > 0:
    return f(*args[:num_args], **fkwargs)
  elif len(fkwargs) > 0:
    return f(**fkwargs)
  return f()


def make_intercept(trace, align_data, align_latent, args, kwargs):
  def _intercept(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    elif align_latent(name) is not None:
      qz = trace[align_latent(name)].value
      if isinstance(qz, RandomVariable):
        value = qz.value
      else:  # e.g. replacement is Tensor
        value = tf.convert_to_tensor(qz)
      fkwargs['value'] = value
    # if auto_transform and 'qz' in locals():
    #   # TODO for generation to work, must output original dist. to
    #   keep around TD? must maintain another stack to write to as a
    #   side-effect (or augment the original stack).
    #   return transform(f, qz, *fargs, **fkwargs)
    return f(*fargs, **fkwargs)
  return _intercept


def transform(f, qz, *args, **kwargs):
  """Transform prior -> unconstrained -> q's constraint.

  When using in VI, we keep variational distribution on its original
  space (for sake of implementing only one intercepting function).
  """
  # TODO deal with f or qz being 'point' or 'points'
  if (not hasattr(f, 'support') or not hasattr(qz, 'support') or
          f.support == qz.support):
    return f(*args, **kwargs)
  value = kwargs.pop('value')
  kwargs['value'] = 0.0  # to avoid sampling; TODO follow sample shape
  rv = f(*args, **kwargs)
  # Take shortcuts in logic if p or q are already unconstrained.
  if qz.support in ('real', 'multivariate_real'):
    return _transform(rv, value=value)
  if rv.support in ('real', 'multivariate_real'):
    rv_unconstrained = rv
  else:
    rv_unconstrained = _transform(rv, value=0.0)
  unconstrained_to_constrained = tfb.Invert(_transform(qz).bijector)
  return _transform(rv_unconstrained,
                    unconstrained_to_constrained,
                    value=value)


def transform(x, *args, **kwargs):
  """Transform a continuous random variable to the unconstrained space.

  `transform` selects among a number of default transformations which
  depend on the support of the provided random variable:

  + $[0, 1]$ (e.g., Beta): Inverse of sigmoid.
  + $[0, \infty)$ (e.g., Gamma): Inverse of softplus.
  + Simplex (e.g., Dirichlet): Inverse of softmax-centered.
  + $(-\infty, \infty)$ (e.g., Normal, MultivariateNormalTriL): None.

  Args:
    x: RandomVariable.
      Continuous random variable to transform.
    *args, **kwargs:
      Arguments to overwrite when forming the `TransformedDistribution`.
      For example, manually specify the transformation by passing in
      the `bijector` argument.

  Returns:
    RandomVariable.
    A `TransformedDistribution` random variable, or the provided random
    variable if no transformation was applied.

  #### Examples

  ```python
  x = Gamma(1.0, 1.0)
  y = ed.transform(x)
  sess = tf.Session()
  sess.run(y)
  -2.2279539
  ```
  """
  if len(args) != 0 or kwargs.get('bijector', None) is not None:
    return TransformedDistribution(x, *args, **kwargs)

  try:
    support = x.support
  except AttributeError as e:
    msg = """'{}' object has no 'support'
             so cannot be transformed.""".format(type(x).__name__)
    raise AttributeError(msg)

  if support == '01':
    bij = tfb.Invert(tfb.Sigmoid())
    new_support = 'real'
  elif support == 'nonnegative':
    bij = tfb.Invert(tfb.Softplus())
    new_support = 'real'
  elif support == 'simplex':
    bij = tfb.Invert(tfb.SoftmaxCentered(event_ndims=1))
    new_support = 'multivariate_real'
  elif support in ('real', 'multivariate_real'):
    return x
  else:
    msg = "'transform' does not handle supports of type '{}'".format(support)
    raise ValueError(msg)

  new_x = TransformedDistribution(x, bij, *args, **kwargs)
  new_x.support = new_support
  return new_x
