Data
----

Data in Edward is stored as a Python dictionary. It is usually comprised
of strings binded to NumPy arrays such as a key ``'x'`` with value
``np.array([0.23512, 13.2])``.
We detail specifics for each modeling language below.

-  **TensorFlow.** The data carries whatever keys and values the user
   accesses in the user-defined model. Key is a string. Value is a NumPy
   array or TensorFlow tensor.

.. code:: python

  class BetaBernoulli:
    def log_prob(self, xs, zs):
      log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
      log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs['x'], p=p))
                         for p in tf.unpack(zs['p'])])
      return log_lik + log_prior

  model = BetaBernoulli()
  data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

-  **Python.** The data carries whatever keys and values the user
   accesses in the user-defined model. Key is a string. Value is a NumPy
   array or TensorFlow tensor.

.. code:: python

  class BetaBernoulli(PythonModel):
    def _py_log_prob(self, xs, zs):
      xs = xs['x']
      ps = zs['p']
      n_samples = ps.shape[0]
      lp = np.zeros(n_samples, dtype=np.float32)
      for b in range(n_samples):
        lp[b] = beta.logpdf(ps[b, :], a=1.0, b=1.0)
        for n in range(xs.shape[0]):
          lp[b] += bernoulli.logpmf(xs[n], p=ps[b, :])

      return lp

  model = BetaBernoulli()
  data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

-  **PyMC3.** The data binds Theano shared variables, which are used to
   mark the observed PyMC3 random variables, to their realizations. Key
   is a Theano shared variable. Value is a NumPy array or TensorFlow
   tensor.

.. code:: python

  x_obs = theano.shared(np.zeros(1))
  with pm.Model() as pm_model:
    p = pm.Beta('p', 1, 1, transform=None)
    x = pm.Bernoulli('x', p, observed=x_obs)

  model = PyMC3Model(pm_model)
  data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

-  **Stan.** The data is according to the Stan program's data block. Key
   is a string. Value is whatever type is used for the data block.

.. code:: python

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
  model = ed.StanModel(model_code=model_code)
  data = {'N': 10, 'x': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

Reading Data in Edward
^^^^^^^^^^^^^^^^^^^^^^

There are three ways to read data in Edward. They follow the `three ways
to read data in TensorFlow
<https://www.tensorflow.org/versions/r0.9/how_tos/reading_data/index.html>`__.

1. **Preloaded data.** A constant or variable in the TensorFlow graph
   holds all the data.

   This setting is the fastest to work with and is recommended if the
   data fits in memory.

   For inference, pass in the data as a dictionary of NumPy arrays.
   Internally, we will store them in TensorFlow variables to prevent
   copying data more than once in memory. Batch training is available
   by passing in the ``n_minibatch`` argument to inference. (As an example, see
   the `mixture of Gaussians
   <https://github.com/blei-lab/edward/blob/master/examples/mixture_gaussian.py>`__.)

2. **Feeding.** Manual code provides the data when running each step of
   inference.

   This setting provides the most fine-grained control which is useful for experimentation.

   For inference, pass in the data as a dictionary of TensorFlow
   placeholders. The user must manually feed the placeholders at each
   step of inference: initialize via ``inference.initialize()``; then
   in a loop call ``sess.run(inference.train, feed_dict={...})`` where
   in ``feed_dict`` you pass in the values for the
   ``tf.placeholder``'s.
   (As an example, see
   the `mixture density network
   <https://github.com/blei-lab/edward/blob/master/examples/mixture_density_network.py>`__
   or `variational auto-encoder
   <https://github.com/blei-lab/edward/blob/master/examples/convolutional_vae.py>`__.)

3. **Reading from files.** An input pipeline reads the data from files
   at the beginning of a TensorFlow graph.

   This setting is recommended if the data does not fit in memory.

   For inference, pass in the data as a dictionary of TensorFlow
   tensors, where the tensors are the output of data readers. (As an
   example, see
   the `data unit test
   <https://github.com/blei-lab/edward/blob/master/tests/test_inference_data.py>`__.)

Training Models with Data
^^^^^^^^^^^^^^^^^^^^^^^^^

How do we use the data during training? In general there are three use
cases:

1. Train over the full data per step.

   Follow the setting of preloaded data.

2. Train over a batch per step when the full data fits in memory. This
   scale inference in terms of computational complexity.

   Follow the setting of preloaded data. Specify the batch size with
   ``n_minibatch`` in ``Inference``. By default, we will subsample by
   slicing along the first dimension of every data structure in the
   data dictionary. Alternatively, follow the setting of feeding.
   Manually deal with the batch behavior at each training step.

3. Train over batches per step when the full data does not fit in
   memory. This scales inference in terms of computational complexity and
   memory complexity.

   Follow the setting of reading from files. Alternatively, follow the
   setting of feeding, and use a generator to create and destroy NumPy
   arrays on the fly for feeding the placeholders.

The three use cases are supported for all modeling languages except
Stan, which is limited to training over the full data per step. (This
because Stan's data structure requires data subsampling on arbitrary
data types, which we don't know how to automate.)
