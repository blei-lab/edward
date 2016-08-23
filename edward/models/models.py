from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.util import get_dims, get_session

try:
  import pystan
  from collections import OrderedDict
except ImportError:
  pass


class PyMC3Model(object):
  """Model wrapper for models written in PyMC3.
  """
  def __init__(self, model):
    """
    Parameters
    ----------
    model : pymc3.Model
      The probability model, written with Theano shared
      variables to form any observations and with
      `transform=None` for any latent variables. The Theano
      shared variables are set during inference, and all latent
      variables live on their original (constrained) space.
    """
    self.model = model
    self.n_vars = None

  def log_prob(self, xs, zs):
    """
    Parameters
    ----------
    xs : dict of str to tf.Tensor
      Data dictionary. Each key is a data structure used in the
      model (Theano shared variable), and its value is the
      corresponding realization (tf.Tensor).
    zs : dict of str to tf.Tensor
      Latent variable dictionary. Each key names a latent variable
      used in the model (str), and its value is the corresponding
      realization (tf.Tensor).

    Returns
    -------
    tf.Tensor
      A 1-D tensor of type tf.float32,
      [log p(xs, zs[1,:]), .., log p(xs, zs[S,:])].

    Notes
    -----
    It wraps around a Python function. The Python function takes
    inputs of type np.ndarray and outputs a np.ndarray.
    """
    # Store keys so that ``_py_log_prob_args`` knows how each
    # value corresponds to a key.
    self.xs_keys = list(six.iterkeys(xs))
    self.zs_keys = list(six.iterkeys(zs))

    # Pass in all tensors as a flattened list for tf.py_func().
    inputs = [tf.convert_to_tensor(x) for x in six.itervalues(xs)]
    inputs += [tf.convert_to_tensor(z) for z in six.itervalues(zs)]

    return tf.py_func(self._py_log_prob_args, inputs, [tf.float32])[0]

  def _py_log_prob_args(self, *args):
    xs_values = args[:len(self.xs_keys)]
    zs_values = args[len(self.xs_keys):]

    # Set data placeholders in PyMC3 model (Theano shared
    # variable) to their realizations (NumPy array).
    for key, value in zip(self.xs_keys, xs_values):
      key.set_value(value)

    # Calculate model's log density using a dictionary of latent
    # variables, one for each sample of latent variables.
    n_samples = zs_values[0].shape[0]
    lp = np.zeros(n_samples, dtype=np.float32)
    for s in range(n_samples):
      z = {key: np.squeeze(value[s, :])
           for key, value in zip(self.zs_keys, zs_values)}
      lp[s] = self.model.fastlogp(z)

    return lp


class PythonModel(object):
  """Model wrapper for models written in NumPy/SciPy.
  """
  def __init__(self):
    self.n_vars = None

  def log_prob(self, xs, zs):
    """
    Parameters
    ----------
    xs : dict of str to tf.Tensor
      Data dictionary. Each key names a data structure used in
      the model (str), and its value is the corresponding
      corresponding realization (tf.Tensor).
    zs : dict of str to tf.Tensor
      Latent variable dictionary. Each key names a latent
      variable used in the model (str), and its value is the
      corresponding realization (tf.Tensor).

    Returns
    -------
    tf.Tensor
      A 1-D tensor of type tf.float32,
      [log p(xs, zs[1,:]), .., log p(xs, zs[S,:])].

    Notes
    -----
    It wraps around a Python function. The Python function takes
    inputs of type np.ndarray and outputs a np.ndarray.
    """
    # Store keys so that ``_py_log_prob_args`` knows how each
    # value corresponds to a key.
    self.xs_keys = list(six.iterkeys(xs))
    self.zs_keys = list(six.iterkeys(zs))

    # Pass in all tensors as a flattened list for tf.py_func().
    inputs = [tf.convert_to_tensor(x) for x in six.itervalues(xs)]
    inputs += [tf.convert_to_tensor(z) for z in six.itervalues(zs)]

    return tf.py_func(self._py_log_prob_args, inputs, [tf.float32])[0]

  def _py_log_prob_args(self, *args):
    # Convert from flattened list to dictionaries for use in a
    # Python function which works with Numpy arrays.
    xs_values = args[:len(self.xs_keys)]
    zs_values = args[len(self.xs_keys):]
    xs = {key: value for key, value in zip(self.xs_keys, xs_values)}
    zs = {key: value for key, value in zip(self.zs_keys, zs_values)}
    return self._py_log_prob(xs, zs)

  def _py_log_prob(self, xs, zs):
    raise NotImplementedError()


class StanModel(object):
  """Model wrapper for models written in Stan.
  """
  def __init__(self, model=None, *args, **kwargs):
    """
    Parameters
    ----------
    model : pystan.StanModel, optional
      An already compiled Stan model. This is useful to avoid
      recompilation of Stan models both within a session (using
      this argument) and across sessions (by loading a pickled
      pystan.StanModel object and passing it in here).
      Alternatively, one can also pickle the ed.StanModel object
      altogether.
    *args
      Passed into pystan.StanModel.
    **kwargs
      Passed into pystan.StanModel.
    """
    if model is None:
      self.model = pystan.StanModel(*args, **kwargs)
    else:
      self.model = model

    self.modelfit = None
    self.is_initialized = False
    self.n_vars = None

  def log_prob(self, xs, zs):
    """
    Parameters
    ----------
    xs : dict
      Data dictionary. Following the Stan program's data block,
      each key names a data structure used in the model (str),
      and its value is the corresponding corresponding
      realization (type is whatever the data block says).
    zs : dict of str to tf.Tensor
      Latent variable dictionary. Following the Stan program's
      parameter block, each key names a latent variable used in
      the model (str), and its value is the corresponding
      realization (tf.Tensor).

    Returns
    -------
    tf.Tensor
      A 1-D tensor of type tf.float32,
      [log p(xs, zs[1,:]), .., log p(xs, zs[S,:])].

    Notes
    -----
    It wraps around a Python function. The Python function takes
    inputs of type np.ndarray and outputs a np.ndarray.
    """
    print("The empty sampling message exists for accessing "
          "Stan's log_prob method.")
    self.modelfit = self.model.sampling(data=xs, iter=1, chains=1)
    if not self.is_initialized:
      self._initialize()

    # Store keys so that ``_py_log_prob_args`` knows how each
    # value corresponds to a key.
    self.zs_keys = list(six.iterkeys(zs))

    # Pass in all tensors as a flattened list for tf.py_func().
    inputs = [tf.convert_to_tensor(z) for z in six.itervalues(zs)]

    return tf.py_func(self._py_log_prob_args, inputs, [tf.float32])[0]

  def _initialize(self):
    self.is_initialized = True
    self.n_vars = sum([sum(dim) if sum(dim) != 0 else 1
                       for dim in self.modelfit.par_dims])

  def _py_log_prob_args(self, *args):
    zs_values = args
    # Calculate model's log density using a dictionary of latent
    # variables, one for each sample of latent variables.
    n_samples = zs_values[0].shape[0]
    lp = np.zeros(n_samples, dtype=np.float32)
    for s in range(n_samples):
      z = {key: np.squeeze(value[s, :])
           for key, value in zip(self.zs_keys, zs_values)}
      # Convert latent variable dictionary (on constrained
      # space) to a flattened vector (on unconstrained space).
      # This is necessary for Stan's log_prob() method.
      z_unconst = self.modelfit.unconstrain_pars(z)
      lp[s] = self.modelfit.log_prob(z_unconst, adjust_transform=False)

    return lp
