from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from edward.models.random_variable import RandomVariable
from tensorflow.contrib import distributions

# Automatically generate random variable classes from classes in
# tf.contrib.distributions.
_globals = globals()
for _name in sorted(dir(distributions)):
  _candidate = getattr(distributions, _name)
  if (inspect.isclass(_candidate) and
          _candidate != distributions.Distribution and
          issubclass(_candidate, distributions.Distribution)):

    params = {'__doc__': _candidate.__doc__}
    _globals[_name] = type(_name, (RandomVariable, _candidate), params)

    del _candidate

# Add supports; these are used, e.g., in conjugacy.
Bernoulli.support = 'binary'
Categorical.support = 'onehot'
Beta.support = '01'
Dirichlet.support = 'simplex'
Gamma.support = 'nonnegative'
InverseGamma.support = 'nonnegative'
MultivariateNormalDiag.support = 'multivariate_real'
Normal.support = 'real'
