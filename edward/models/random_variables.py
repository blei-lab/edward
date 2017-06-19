from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect as _inspect

from edward.models.random_variable import RandomVariable as _RandomVariable
from tensorflow.contrib import distributions as _distributions

# Automatically generate random variable classes from classes in
# tf.contrib.distributions.
_globals = globals()
for _name in sorted(dir(_distributions)):
  _candidate = getattr(_distributions, _name)
  if (_inspect.isclass(_candidate) and
          _candidate != _distributions.Distribution and
          issubclass(_candidate, _distributions.Distribution)):

    # to use _candidate's docstring, must write a new __init__ method
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
