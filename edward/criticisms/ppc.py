from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from edward.models import RandomVariable
from edward.util import get_session


def ppc(T, data, latent_vars=None, model_wrapper=None, n_samples=100):
  """Posterior predictive check
  (Rubin, 1984; Meng, 1994; Gelman, Meng, and Stern, 1996).

  If ``latent_vars`` is inputted as ``None``, then it is a prior
  predictive check (Box, 1980).

  PPC's form an empirical distribution for the predictive discrepancy,

  .. math::
    p(T) = \int p(T(x^{rep}) | z) p(z | x) dz

  by drawing replicated data sets xrep and calculating
  :math:`T(x^{rep})` for each data set. Then it compares it to
  :math:`T(x)`.

  Parameters
  ----------
  T : function
    Discrepancy function, which takes a dictionary of data and
    dictionary of latent variables as input and outputs a ``tf.Tensor``.
  data : dict
    Data to compare to. It binds observed variables (of type
    ``RandomVariable``) to their realizations (of type ``tf.Tensor``). It
    can also bind placeholders (of type ``tf.Tensor``) used in the model
    to their realizations.
  latent_vars : dict of str to RandomVariable, optional
    Collection of random variables binded to their inferred posterior.
    It is an optional argument, necessary for when the discrepancy is
    a function of latent variables.
  model_wrapper : ed.Model, optional
    An optional wrapper for the probability model. It must have a
    ``sample_likelihood`` method. If ``latent_vars`` is not specified,
    it must also have a ``sample_prior`` method, as ``ppc`` will
    default to a prior predictive check. ``data`` is also changed. For
    TensorFlow, Python, and Stan models, the key type is a string; for
    PyMC3, the key type is a Theano shared variable. For TensorFlow,
    Python, and PyMC3 models, the value type is a NumPy array or
    TensorFlow placeholder; for Stan, the value type is the type
    according to the Stan program's data block.
  n_samples : int, optional
    Number of replicated data sets.

  Returns
  -------
  list of np.ndarray
    List containing the reference distribution, which is a NumPy
    array of size elements,

    .. math::
      (T(x^{rep,1}, z^{1}), ..., T(x^{rep,size}, z^{size}))

    and the realized discrepancy, which is a NumPy array of size
    elements,

    .. math::
      (T(x, z^{1}), ..., T(x, z^{size})).


  Examples
  --------
  >>> # build posterior predictive after inference: it is
  >>> # parameterized by posterior means
  >>> x_post = copy(x, {z: qz.mean(), beta: qbeta.mean()})
  >>>
  >>> # posterior predictive check
  >>> # T is a user-defined function of data, T(data)
  >>> T = lambda xs, zs: tf.reduce_mean(xs[x_post])
  >>> ppc(T, data={x_post: x_train})
  >>>
  >>> # in general T is a discrepancy function of the data (both response and
  >>> # covariates) and latent variables, T(data, latent_vars)
  >>> T = lambda xs, zs: tf.reduce_mean(zs['z'])
  >>> ppc(T, data={y_post: y_train, x_ph: x_train},
  ...     latent_vars={'z': qz, 'beta': qbeta})
  >>>
  >>> # prior predictive check
  >>> # running ppc on original x
  >>> ppc(T, data={x: x_train})
  """
  sess = get_session()
  # Sample to get replicated data sets and latent variables.
  if model_wrapper is None:
    if latent_vars is None:
      zrep = None
    else:
      zrep = {key: value.value()
              for key, value in six.iteritems(latent_vars)}

    xrep = {}
    for x, obs in six.iteritems(data):
      if isinstance(x, RandomVariable):
        # Replace observed data with replicated data.
        xrep[x] = x.value()
      else:
        xrep[x] = obs
  else:
    if latent_vars is None:
      zrep = model_wrapper.sample_prior()
    else:
      zrep = {key: value.value()
              for key, value in six.iteritems(latent_vars)}

    xrep = model_wrapper.sample_likelihood(zrep)

  # Create feed_dict for data placeholders that the model conditions
  # on; it is necessary for all session runs.
  feed_dict = {x: obs for x, obs in six.iteritems(data)
               if not isinstance(x, RandomVariable) and
               not isinstance(x, str)}

  # Calculate discrepancy over many replicated data sets and latent
  # variables.
  Trep = T(xrep, zrep)
  Tobs = T(data, zrep)
  Treps = []
  Ts = []
  for s in range(n_samples):
    # Take a forward pass (session run) to get new samples for
    # each calculation of the discrepancy.
    # Note that alternatively we can unroll the graph by registering
    # this operation ``n_samples`` times, each for different parent
    # nodes representing ``xrep`` and ``zrep``.
    Treps += [sess.run(Trep, feed_dict)]
    Ts += [sess.run(Tobs, feed_dict)]

  return [np.stack(Treps), np.stack(Ts)]
