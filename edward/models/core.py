from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect as _inspect

from edward.models.random_variable import RandomVariable as _RandomVariable
from tensorflow.contrib import distributions as _distributions

TRACE_STACK = [lambda f, *args, **kwargs: f(*args, **kwargs)]


def call_with_manipulate(f, manipulate, *args, **kwargs):
  """Calls function `f(*args, **kwargs)` with manipulation.

  Args:
    f: Function to call.
    manipulate: Function to intercept primitives. It takes each primitive
      function `f`, inputs `args, kwargs`, and may return any value and/or add
      side-effects.
    args, kwargs: Inputs to function.

  Returns:
    The output of `f`. Any calls to `primitive` operations are replaced by
    calls to `manipulate`.

  #### Examples

  ```python
  def f(x):
    y = Poisson(rate=x, name="y")
    return y

  def manipulate(f, *args, **kwargs):
    if kwargs.get("name") == "y":
      kwargs["value"] = 42
    return f(*args, **kwargs)

  y = ed.call_with_manipulate(f, manipulate, 1.5)
  with tf.Session() as sess:
    assert sess.run(y.value) == 42
  ```
  """
  TRACE_STACK.append(manipulate)
  output = f(*args, **kwargs)
  TRACE_STACK.pop()
  return output


def primitive(cls_init):
  """Wraps class __init__ for manipulating its continuation."""
  def __init__(self, *args, **kwargs):
    TRACE_STACK[-1](cls_init, self, *args, **kwargs)
  return __init__


# Automatically generate random variable classes from classes in
# tf.contrib.distributions.
_globals = globals()
for _name in sorted(dir(_distributions)):
  _candidate = getattr(_distributions, _name)
  if (_inspect.isclass(_candidate) and
          _candidate != _distributions.Distribution and
          issubclass(_candidate, _distributions.Distribution)):

    # write a new __init__ method in order to decorate class as primitive
    # and share _candidate's docstring
    @primitive
    def __init__(self, *args, **kwargs):
      _RandomVariable.__init__(self, *args, **kwargs)
    __init__.__doc__ = _candidate.__init__.__doc__
    _params = {'__doc__': _candidate.__doc__,
               '__init__': __init__}
    _globals[_name] = type(_name, (_RandomVariable, _candidate), _params)

    del _candidate

# Add supports; these are used, e.g., in conjugacy.
Bernoulli.support = 'binary'
Beta.support = '01'
Binomial.support = 'onehot'
Categorical.support = 'categorical'
Chi2.support = 'nonnegative'
Dirichlet.support = 'simplex'
Exponential.support = 'nonnegative'
Gamma.support = 'nonnegative'
InverseGamma.support = 'nonnegative'
Laplace.support = 'real'
Multinomial.support = 'onehot'
MultivariateNormalDiag.support = 'multivariate_real'
Normal.support = 'real'
Poisson.support = 'countable'

del absolute_import
del division
del print_function
