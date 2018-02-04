from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import operator
import six
import tensorflow as tf

from edward.models.core import call_with_manipulate
from edward.models.core import TransformedDistribution
from edward.models.random_variable import RandomVariable

tfb = tf.contrib.distributions.bijectors


def call_with_trace(f, *args, **kwargs):
  """Calls function and writes to a stack to expose its execution trace."""
  def manipulate(cls_init, self, *fargs, **fkwargs):
    cls_init(self, *fargs, **fkwargs)
    stack[self.name] = self
  stack = collections.OrderedDict({})
  f = make_optional_inputs(f)
  call_with_manipulate(f, manipulate, *args, **kwargs)
  return stack


def call_with_intercept(f, trace, align_data, align_latent,
                        *args, **kwargs):
  """Calls function and intercepts its primitive ops' sample values."""
  def manipulate(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    elif align_latent(name) is not None:
      fkwargs['value'] = tf.convert_to_tensor(trace[align_latent(name)])
    # if auto_transform and 'qz' in locals():
    #   # TODO for generation to work, must output original dist. to
    #   keep around TD? must maintain another stack to write to as a
    #   side-effect (or augment the original stack).
    #   return transform(f, qz, *fargs, **fkwargs)
    return f(*fargs, **fkwargs)
  f = make_optional_inputs(f)
  return call_with_manipulate(f, manipulate, *args, **kwargs)


def make_log_joint(model, states):
  """Factory to make a log-joint probability function.

  It takes a model and transition states as input. It returns its log-joint
  probability as a function of the states. (This is applied in Markov chain
  Carlo algorithms.)
  """
  maybe_list = lambda x: list(x) if isinstance(x, (tuple, list)) else [x]
  states = maybe_list(states)
  def log_joint(*fargs):
    """Target's unnormalized log-joint density as a function of states."""
    q_trace = {state.name.split(':')[0]: arg
               for state, arg in zip(states, fargs)}
    x = call_with_intercept(model, q_trace, align_data, align_latent,
                            *args, **kwargs)
    p_log_prob = 0.0
    for rv in toposort(x):
      if align_latent(rv.name) is not None or align_data(rv.name) is not None:
        p_log_prob += tf.reduce_sum(rv.log_prob(rv.value))
    return p_log_prob
  return log_joint


def make_optional_inputs(f):
  """Wraps function to take in optional, unused args/kwargs."""
  def f_wrapped(*args, **kwargs):
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
  f_wrapped.__name__ = getattr(f, '__name__', '[unknown name]')
  f_wrapped.__doc__ = getattr(f, '__doc__' , '')
  return f_wrapped


def toposort(end_node, parents=operator.methodcaller('get_parents')):
  """Generate nodes in DAG's reverse topological order.

  For any edge U -> V, the function visits V before visiting U. It traces
  using a backward pass, i.e., the "pull" dataflow model.

  Args:
    end_node: Input or list of inputs.
  """
  child_counts = {}
  maybe_list = lambda x: list(x) if isinstance(x, (list, tuple)) else [x]
  stack = maybe_list(end_node)
  while stack:
    node = stack.pop()
    if node in child_counts:
      child_counts[node] += 1
    else:
      child_counts[node] = 1
      stack.extend(parents(node))

  childless_nodes = maybe_list(end_node)
  while childless_nodes:
    node = childless_nodes.pop()
    yield node
    for parent in parents(node):
      if child_counts[parent] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[parent] -= 1


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


def get_control_variate_coef(f, h):
  """Returns scalar used by control variates method for variance reduction in
  Monte Carlo methods.

  If we have a statistic $m$ as an unbiased estimator of $\mu$ and
  and another statistic $t$ which is an unbiased estimator of
  $\\tau$ then $m^* = m + c(t - \\tau)$ is also an unbiased
  estimator of $\mu$ for any coefficient $c$.

  This function calculates the optimal coefficient

  $c^* = \\frac{\\text{Cov}(m,t)}{\\text{Var}(t)}$

  for minimizing the variance of $m^*$.

  Args:
    f: tf.Tensor.
      A 1-D tensor.
    h: tf.Tensor.
      A 1-D tensor.

  Returns:
    tf.Tensor.
    A 0 rank tensor
  """
  f_mu = tf.reduce_mean(f)
  h_mu = tf.reduce_mean(h)

  n = f.shape[0].value

  cov_fh = tf.reduce_sum((f - f_mu) * (h - h_mu)) / (n - 1)
  var_h = tf.reduce_sum(tf.square(h - h_mu)) / (n - 1)

  a = cov_fh / var_h

  return a
