Building Probabilistic Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A probabilistic model is specified by a joint distribution ``p(x,z)`` of
data ``x`` and latent variables ``z``. All models in Edward are written
as a class; to implement a new model, it can be written in any of the
currently supported modeling languages: TensorFlow, Python, Stan, and
PyMC3.

To use TensorFlow or Python, write a class with the method
``log_prob(xs, zs)``. The method defines the logarithm of a joint
probabilistic density function over data ``xs`` and latent variables
``zs``. ``xs`` can be a single data point or a batch, and analogously,
``zs`` can be a single set or multiple sets of latent variables. The
method outputs a vector of the joint density evaluations
``[log p(xs, zs[0,:]), log p(xs, zs[1,:]), ...]``, with an evaluation
for each set of latent variables. Here is an example:

.. code:: python

    class BetaBernoulli:
        """
        p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
        """
        def log_prob(self, xs, zs):
            log_prior = beta.logpdf(zs, a=1.0, b=1.0)
            log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs['x'], z))
                               for z in tf.unpack(zs)])
            return log_lik + log_prior

    model = BetaBernoulli()

Here is a `toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_tf.py>`__
that uses this model which is written in TensorFlow. Here is another
`toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_np.py>`__
that uses the same model written in Python using only NumPy/SciPy. The
model class can be more complicated, containing fields or other methods
required for certain functions in Edward. A full spec of the model is
available below. [Include the full model spec somewhere below.]

To use PyMC3, write a PyMC3 model whose observed values are Theano
shared variables. The values in the Theano shared variables can be
plugged at a later time. Here is an example:

.. code:: python

    x_obs = theano.shared(np.zeros(1))
    with pm.Model() as pm_model:
        beta = pm.Beta('beta', 1, 1, transform=None)
        x = pm.Bernoulli('x', beta, observed=x_obs)

Here is a `toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_pymc3.py>`__
that uses this model.

To use Stan, write a Stan program in the form of a file or string. Then
call it with ``StanModel(file)`` or ``StanModel(model_code)``. Here is
an example:

.. code:: python

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
    model = ed.StanModel(model_code=model_code)

Here is a `toy
script <https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_stan.py>`__
that uses this model. Stan programs are convenient as `there are many
online examples <https://github.com/stan-dev/example-models/wiki>`__,
although they are limited to probability models with differentiable
latent variables and they can be quite slow to call in practice over
TensorFlow.

For efficiency during future inferences or criticisms, we recommend
using the modeling language which contains the most structure about the
model; this enables the inference algorithms to automatically take
advantage of any available structure if they are implemented to do so.
TensorFlow will be most efficient as Edward uses it as the backend for
computation. Internally, we wrap the other languages in TensorFlow so
their computation represents a single node in the graph (making it more
difficult to tease apart and thus distribute).
