from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as _collections
import inspect as _inspect

from edward.models.random_variable import RandomVariable as _RandomVariable
from tensorflow.contrib import distributions as _distributions


class Node(object):
  """Node in execution trace. A trace's nodes form a directed acyclic graph."""
  __slots__ = ['value', 'f', 'args', 'kwargs', 'parents']

  def __init__(self, value, f, args, kwargs, parents):
    self.value = value
    self.f = f
    self.args = args
    self.kwargs = kwargs
    self.parents = parents


def primitive(cls_init):
  """Wraps class __init__ for recording and intercepting."""
  def __init__(self, *args, **kwargs):
    global _INTERCEPT, _STORE_ARGS, _TRACE_STACK
    if '_INTERCEPT' in globals() and callable(_INTERCEPT):
      _INTERCEPT(cls_init, self, *args, **kwargs)
    else:
      cls_init(self, *args, **kwargs)
    if '_STORE_ARGS' in globals() and '_TRACE_STACK' in globals():
      if _STORE_ARGS:
        parents = [v for v in list(args) + kwargs.values()
                   if hasattr(v, "name") and v.name in _TRACE_STACK]
        _TRACE_STACK[self.name] = Node(self, cls_init, args, kwargs, parents)
      else:
        _TRACE_STACK[self.name] = Node(self, None, None, None, None)
  return __init__


def trace(f, *args, **kwargs):
  """Traces the function `f(*args, **kwargs)`.

  Args:
    f: Function to trace.
    intercept: Function to intercept primitives. It takes a primitive
      function `f`, inputs `args, kwargs`, and may return any value and/or
      add side-effects. Default is `None`, equivalent to `f(*args, **kwargs)`.
    store_args: Boolean for whether `Node`s store their inputs and parent
      primitives. Default is `False`.
    args, kwargs: (Possible) inputs to function.

  Returns:
    The execution trace of `f`, collecting any `primitive` operations that the
    function executed. It is reified as a stack (`OrderedDict`), and each
    executed primitive is a `Node` on the stack indexed by its string name.

  #### Examples

  ```python
  def f(x):
    y = Poisson(rate=x, name="y")

  def intercept(f, *args, **kwargs):
    if kwargs.get("name") == "y":
      kwargs["value"] = 42
    return f(*args, **kwargs)

  trace_stack = ed.trace(f, 1.5, intercept=intercept)
  print(trace_stack)
  ## OrderedDict([('y', <edward.models.core.Node object at 0x118c1ce10>)])

  rv = trace_stack["y"].value
  with tf.Session() as sess:
    assert sess.run(rv.value) == 42
  ```
  """
  # TODO move call_function_up_to_args
  from edward.inferences.util import call_function_up_to_args
  global _INTERCEPT, _STORE_ARGS, _TRACE_STACK
  _INTERCEPT = kwargs.pop("intercept", None)
  _STORE_ARGS = kwargs.pop("store_args", False)
  _TRACE_STACK = _collections.OrderedDict({})
  call_function_up_to_args(f, *args, **kwargs)
  output = _TRACE_STACK
  del _INTERCEPT, _STORE_ARGS, _TRACE_STACK
  return output


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
