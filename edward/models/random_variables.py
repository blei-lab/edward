from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tensorflow as tf

from edward.models.empirical import Empirical as distributions_Empirical
from edward.models.point_mass import PointMass as distributions_PointMass
from edward.models.random_variable import RandomVariable
from tensorflow.contrib import distributions


class Empirical(RandomVariable, distributions_Empirical):
  def __init__(self, *args, **kwargs):
    super(Empirical, self).__init__(*args, **kwargs)


class PointMass(RandomVariable, distributions_PointMass):
  def __init__(self, *args, **kwargs):
    super(PointMass, self).__init__(*args, **kwargs)


# Automatically generate random variable classes from classes in
# tf.contrib.distributions.
_globals = globals()
for _name in sorted(dir(distributions)):
  _candidate = getattr(distributions, _name)
  if (inspect.isclass(_candidate) and
          _candidate != distributions.Distribution and
          issubclass(_candidate, distributions.Distribution)):

    class _WrapperRandomVariable(RandomVariable, _candidate):
      def __init__(self, *args, **kwargs):
        RandomVariable.__init__(self, *args, **kwargs)

    _WrapperRandomVariable.__name__ = _name
    _globals[_name] = _WrapperRandomVariable

    del _WrapperRandomVariable
    del _candidate
