Probabilistic Models
^^^^^^^^^^^^^^^^^^^^

A probabilistic model specifies a joint distribution ``p(x, z)``
of data ``x`` and latent variables ``z``.
For more details, see the
`Probability Models tutorial <>`__.

All models in Edward are written as a class. To write a new model,
it can be written in any of the currently supported modeling
languages: TensorFlow, Python, Stan, and PyMC3.

**TensorFlow.**
Write a class with the method ``log_prob(xs, zs)``. The method defines
the logarithm of a joint density. ``xs`` can be a single data point or
a batch, and analogously, ``zs`` can be a single set or multiple sets
of latent variables. The method outputs a vector of the joint density
evaluations ``[log p(xs, zs[0,:]), log p(xs, zs[1,:]), ...]``, with an
evaluation for each set of latent variables. Here is an example:

.. code:: python

    class BetaBernoulli:
        """p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)"""
        def log_prob(self, xs, zs):
            log_prior = beta.logpdf(zs, a=1.0, b=1.0)
            log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs['x'], z))
                               for z in tf.unpack(zs)])
            return log_lik + log_prior

    model = BetaBernoulli()

Here is a `toy script
<https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_tf.py>`__
that uses this model. The model class can be more complicated,
containing fields or other methods required for certain functions in
Edward, and which can provide more information about the model's
structure. A full spec of the model is available below. [Include the
full model spec somewhere below.]

**Python.**
Write a class that inherits from ``PythonModel`` and with the method
``_py_log_prob(xs, zs)``. The method defines the logarithm of a joint
density with the same concept as in a TensorFlow model, but where
``xs`` and ``zs`` now use NumPy arrays rather than TensorFlow tensors.
Here is an example:

.. code:: python

  from edward.models import PythonModel

  class BetaBernoulli(PythonModel):
      """p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)"""
      def _py_log_prob(self, xs, zs):
          # This example is written for pedagogy. We recommend
          # vectorizing operations in practice.
          n_samples = zs.shape[0]
          lp = np.zeros(n_samples, dtype=np.float32)
          for b in range(n_samples):
              lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
              for n in range(xs['x'].shape[0]):
                  lp[b] += bernoulli.logpmf(xs['x'][n], p=zs[b, :])

          return lp
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
          int<lower=0,upper=1> y[N];
        }
        parameters {
          real<lower=0,upper=1> theta;
        }
        model {
          theta ~ beta(1.0, 1.0);
          for (n in 1:N)
            y[n] ~ bernoulli(theta);
        }
    """
    model = StanModel(model_code=model_code)

Here is a `toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_stan.py>`__
that uses this model. Stan programs are convenient as `there are many
online examples <https://github.com/stan-dev/example-models/wiki>`__,
although they are limited to probability models with differentiable
latent variables. ``StanModel`` objects also contain no structure about
the model besides how to calculate its joint density.

**PyMC3.**
Write a PyMC3 model whose observed values are Theano shared variables.
The values in the Theano shared variables can be plugged at a later
time. Here is an example:

.. code:: python

    from edward.models import PyMC3Model

    x_obs = theano.shared(np.zeros(1))
    with pm.Model() as pm_model:
        beta = pm.Beta('beta', 1, 1, transform=None)
        x = pm.Bernoulli('x', beta, observed=x_obs)

    model = PyMC3Model(pm_model)

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
`tutorials <>`__.
