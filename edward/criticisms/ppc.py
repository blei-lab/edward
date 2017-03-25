from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable
from edward.util import check_data, check_latent_vars, get_session


def ppc(T, data, latent_vars=None, n_samples=100):
  """Posterior predictive check
  (Rubin, 1984; Meng, 1994; Gelman, Meng, and Stern, 1996).

  PPC's form an empirical distribution for the predictive discrepancy,

  .. math::
    p(T\mid x) = \int p(T(x^{\\text{rep}})\mid z) p(z\mid x) dz

  by drawing replicated data sets :math:`x^{\\text{rep}}` and
  calculating :math:`T(x^{\\text{rep}})` for each data set. Then it
  compares it to :math:`T(x)`.

  If ``data`` is inputted with the prior predictive distribution, then
  it is a prior predictive check (Box, 1980).

  Parameters
  ----------
  T : function
    Discrepancy function, which takes a dictionary of data and
    dictionary of latent variables as input and outputs a ``tf.Tensor``.
  data : dict
    Data to compare to. It binds observed variables (of type
    ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
    type ``tf.Tensor``). It can also bind placeholders (of type
    ``tf.Tensor``) used in the model to their realizations.
  latent_vars : dict, optional
    Collection of random variables (of type ``RandomVariable`` or
    ``tf.Tensor``) binded to their inferred posterior. This argument
    is used when the discrepancy is a function of latent variables.
  n_samples : int, optional
    Number of replicated data sets.

  Returns
  -------
  list of np.ndarray
    List containing the reference distribution, which is a NumPy array
    with ``n_samples`` elements,

    .. math::
      (T(x^{{\\text{rep}},1}, z^{1}), ...,
       T(x^{\\text{rep,nsamples}}, z^{\\text{nsamples}}))

    and the realized discrepancy, which is a NumPy array with
    ``n_samples`` elements,

    .. math::
      (T(x, z^{1}), ..., T(x, z^{\\text{nsamples}})).


  Examples
  --------
  >>> # build posterior predictive after inference:
  >>> # it is parameterized by a posterior sample
  >>> x_post = ed.copy(x, {z: qz, beta: qbeta})
  >>>
  >>> # posterior predictive check
  >>> # T is a user-defined function of data, T(data)
  >>> T = lambda xs, zs: tf.reduce_mean(xs[x_post])
  >>> ed.ppc(T, data={x_post: x_train})
  >>>
  >>> # in general T is a discrepancy function of the data (both response and
  >>> # covariates) and latent variables, T(data, latent_vars)
  >>> T = lambda xs, zs: tf.reduce_mean(zs[z])
  >>> ed.ppc(T, data={y_post: y_train, x_ph: x_train},
  ...        latent_vars={z: qz, beta: qbeta})
  >>>
  >>> # prior predictive check
  >>> # run ppc on original x
  >>> ed.ppc(T, data={x: x_train})
  """
  sess = get_session()
  if not callable(T):
    raise TypeError("T must be a callable function.")

  check_data(data)
  if latent_vars is None:
    latent_vars = {}

  check_latent_vars(latent_vars)
  if not isinstance(n_samples, int):
    raise TypeError("n_samples must have type int.")

  # Build replicated latent variables.
  zrep = {key: tf.convert_to_tensor(value)
          for key, value in six.iteritems(latent_vars)}

  # Build replicated data.
  xrep = {x: (x.value() if isinstance(x, RandomVariable) else obs)
          for x, obs in six.iteritems(data)}

  # Create feed_dict for data placeholders that the model conditions
  # on; it is necessary for all session runs.
  feed_dict = {key: value for key, value in six.iteritems(data)
               if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type}

  # Calculate discrepancy over many replicated data sets and latent
  # variables.
  Trep = T(xrep, zrep)
  Tobs = T(data, zrep)
  Treps = []
  Ts = []
  for _ in range(n_samples):
    # Take a forward pass (session run) to get new samples for
    # each calculation of the discrepancy.
    # Alternatively, we could unroll the graph by registering this
    # operation ``n_samples`` times, each for different parent nodes
    # representing ``xrep`` and ``zrep``. But it's expensive.
    Treps += [sess.run(Trep, feed_dict)]
    Ts += [sess.run(Tobs, feed_dict)]

  return [np.stack(Treps), np.stack(Ts)]
