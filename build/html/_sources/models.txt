Models
------

A probabilistic model specifies a joint distribution ``p(x, z)``
of data ``x`` and latent variables ``z``.
For more details, see the
`Probability Models tutorial <../tut_model>`__.

All models in Edward are written as a class. To write a new model,
it can be written in any of the currently supported modeling
languages: TensorFlow, Python, Stan, and PyMC3.

**TensorFlow.**
Write a class with the method ``log_prob(xs, zs)``. The method defines
the logarithm of a joint density, where ``xs`` and ``zs`` are Python
dictionaries binding the name of a random variable to
a realization.
Here ``xs`` can be a single data
point or a batch of data points, and analogously, ``zs`` can be a
single set or multiple sets of latent variables.
Here is an example:

.. code:: python

  import tensorflow as tf
  from edward.stats import bernoulli, beta

  class BetaBernoulli:
    """p(x, p) = Bernoulli(x | p) * Beta(p | 1, 1)"""
    def log_prob(self, xs, zs):
      log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
      log_lik = tf.reduce_sum(bernoulli.logpmf(xs['x'], p=zs['p']))
      return log_lik + log_prior

  model = BetaBernoulli()

``BetaBernoulli`` defines a log joint density with a Bernoulli
likelihood (for an unspecified number of data points) and a Beta prior
on the Bernoulli's success probability.
``xs`` is a dictionary with string ``x`` binded to a vector of
observations. ``zs`` is a dictionary with string ``z`` binded to a
sample from the one-dimensional Beta latent variable.

Here is a `toy script
<https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_tf.py>`__
that uses this model. The model class can be more complicated,
containing fields or other methods required for other functionality in
Edward. See the section below for more details.

**Python.**
Write a class that inherits from ``PythonModel`` and with the method
``_py_log_prob(xs, zs)``. The method defines the logarithm of a joint
density with the same concept as in a TensorFlow model, but where
``xs`` and ``zs`` now use NumPy arrays rather than TensorFlow tensors.
Here is an example:

.. code:: python

  import numpy as np
  from edward.models import PythonModel
  from scipy.stats import bernoulli, beta

  class BetaBernoulli(PythonModel):
    """p(x, p) = Bernoulli(x | p) * Beta(p | 1, 1)"""
    def _py_log_prob(self, xs, zs):
      log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
      log_lik = np.sum(bernoulli.logpmf(xs['x'], p=zs['p']))
      return log_lik + log_prior

    model = BetaBernoulli()

Here is a `toy script
<https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_np.py>`__
that uses this model.

**Stan.**
Write a Stan program in the form of a file or string. Then
call it with ``StanModel(file=file)`` or
``StanModel(model_code=model_code)``. Here is an example:

.. code:: python

  from edward.models import StanModel

  model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> x[N];
    }
    parameters {
      real<lower=0,upper=1> p;
    }
    model {
      p ~ beta(1.0, 1.0);
      for (n in 1:N)
      x[n] ~ bernoulli(p);
    }
  """
  model = StanModel(model_code=model_code)

During inference the latent variable string matches the name of the
parameters from the parameter block. Analogously, the data's string
matches the name of the data from the data block.

.. code:: python

  qp = Beta(...)
  data = {'N': 10, 'x': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
  inference = Inference({'p': qp}, data, model)

Here is a `toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_stan.py>`__
that uses this model. Stan programs are convenient as `there are many
online examples <https://github.com/stan-dev/example-models/wiki>`__,
although they are limited to probability models with differentiable
latent variables. ``StanModel`` objects also contain no structure about
the model besides how to calculate its joint density.

**PyMC3.**
Write a PyMC3 model whose observed values are Theano shared variables,
and whose latent variables use ``transform=None`` to keep them on their
original (constrained) domain.
The values in the Theano shared variables can be plugged at a later
time. Here is an example:

.. code:: python

  import numpy as np
  import pymc3 as pm
  import theano
  from edward.models import PyMC3Model

  x_obs = theano.shared(np.zeros(1))
  with pm.Model() as pm_model:
    p = pm.Beta('p', 1, 1, transform=None)
    x = pm.Bernoulli('x', p, observed=x_obs)

  model = PyMC3Model(pm_model)

During inference the latent variable string matches the name of the
model's latent variables; the data's string matches the Theano shared
variables.

.. code:: python

  qp = Beta(...)
  data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
  inference = Inference({'p': qp}, data, model)

Here is a `toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_pymc3.py>`__
that uses this model. PyMC3 can be used to define models with both
differentiable latent variables and non-differentiable (e.g., discrete)
latent variables. ``PyMC3Model`` objects contain no structure about the
model besides how to calculate its joint density.

For modeling convenience, we recommend using the modeling language that
you are most familiar with. For efficiency, we recommend using
TensorFlow, as Edward uses TensorFlow as the computational backend.
Internally, other languages are wrapped in TensorFlow so their
computation represents a single node in the graph (making it difficult
to tease apart and thus distribute their computation).

For examples of models built in Edward, see the model
`tutorials <../tutorials>`__.

Model API
^^^^^^^^^

This outlines the current spec for all methods in the model object.
It includes all modeling languages, where certain methods are
implemented by wrapping around other methods. For example, by a Python
model builds a ``_py_log_prob()`` method and inherits from
``PythonModel``; ``PythonModel`` implements ``log_prob()`` by wrapping
around ``_py_log_prob()`` as a TensorFlow operation.

.. code:: python

  class Model:
    def log_prob(self, xs, zs):
      """
      Used in: (most) inference.

      Parameters
      ----------
      xs : dict of str to tf.Tensor
        Data dictionary. Each key names a data structure used in the
        model (str), and its value is the corresponding corresponding
        realization (tf.Tensor).
      zs : dict of str to tf.Tensor
        Latent variable dictionary. Each key names a latent variable
        used in the model (str), and its value is the corresponding
        realization (tf.Tensor).

      Returns
      -------
      tf.Tensor
        Scalar, the log joint density log p(xs, zs).
      """
      pass

    def log_lik(self, xs, zs):
      """
      Used in: inference with analytic KL.

      Parameters
      ----------
      xs : dict of str to tf.Tensor
        Data dictionary. Each key names a data structure used in the
        model (str), and its value is the corresponding corresponding
        realization (tf.Tensor).
      zs : dict of str to tf.Tensor
        Latent variable dictionary. Each key names a latent variable
        used in the model (str), and its value is the corresponding
        realization (tf.Tensor).

      Returns
      -------
      tf.Tensor
        Scalar, the log-likelihood log p(xs | zs).
      """

    def predict(self, xs, zs):
      """
      Used in: ed.evaluate().

      Parameters
      ----------
      xs : dict of str to tf.Tensor
        Data dictionary. Each key names a data structure used in the
        model (str), and its value is the corresponding corresponding
        realization (tf.Tensor).
      zs : dict of str to tf.Tensor
        Latent variable dictionary. Each key names a latent variable
        used in the model (str), and its value is the corresponding
        realization (tf.Tensor).

      Returns
      -------
      tf.Tensor
        Tensor of predictions, one for each data point. The prediction
        is the likelihood's mean. For example, in supervised learning
        of i.i.d. categorical data, it is a vector of labels.
      """
      pass

    def sample_prior(self):
      """
      Used in: ed.ppc().

      Returns
      -------
      dict of str to tf.Tensor
        Latent variable dictionary. Each key names a latent variable
        used in the model (str), and its value is the corresponding
        realization (tf.Tensor).
      """
      pass

    def sample_likelihood(self, zs):
      """
      Used in: ed.ppc().

      Parameters
      ----------
      zs : dict of str to tf.Tensor
        Latent variable dictionary. Each key names a latent variable
        used in the model (str), and its value is the corresponding
        realization (tf.Tensor).

      Returns
      -------
      dict of str to tf.Tensor
        Data dictionary. It is a replicated data set, where each key
        and value matches the same type as any observed data set that
        the model aims to capture.
      """
      pass
