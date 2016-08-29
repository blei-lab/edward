Inference
---------

An inference algorithm infers the posterior for a particular model
``p(x, z)`` and data set ``x``. It is the distribution of the latent
variables given data, ``p(z | x)``. For more details, see the
`Inference of Probability Models tutorial <../tut_inference>`__.

Edward uses classes and class inheritance to provide a
hierarchy of inference methods, all of which are easily extensible.
This enables fast experimentation and research on top of existing
inference methods, whether it be developing new black box inference
algorithms or developing new model-specific inference algorithms which
are tailored to a particular model or restricted class of models.
We detail this below.

.. image:: ../images/inference_structure.png

*Dependency graph of inference methods. Nodes are classes in Edward
and arrows represent class inheritance.*

There is a base class ``Inference``, from which all inference
methods are derived from.

.. code:: python

  class Inference(object):
    """Base class for Edward inference methods.
    """
    def __init__(self, latent_vars, data=None, model_wrapper=None):
      ...

It takes as input the set of latent variables to infer and a dataset. For now, it must also take as input a model wrapper ``model_wrapper``.
For more details, see the
`Data API <data>`__.
and
`Model API <models>`__

Note that ``Inference`` says nothing about the class of models that an
algorithm must work with. One can build inference algorithms which are
tailored to a restricted class of models available in Edward (such as
differentiable models), or even tailor it to a single model. The
algorithm can raise an error if the model is outside this class.

We organize inference under two paradigms:
``VariationalInference`` and ``MonteCarlo`` (or more plainly,
optimization and sampling). These inherit from ``Inference`` and each
have their own default methods.

.. code:: python

  class MonteCarlo(Inference):
    """Base class for Monte Carlo inference methods.
    """
    def __init__(latent_vars, data=None, model_wrapper=None):
      super(MonteCarlo, self).__init__(latent_vars, data, model_wrapper)

    ...


  class VariationalInference(Inference):
    """Base class for variational inference methods.
    """
    def __init__(self, latent_vars, data=None, model_wrapper=None):
      """Initialization.
      ...
      """
      super(VariationalInference, self).__init__(latent_vars, data, model_wrapper)

    ...

Hybrid methods and novel paradigms outside of ``VariationalInference``
and ``MonteCarlo`` are also possible in Edward. For example, one can
write a class derived from ``Inference`` directly, or inherit to
carry both ``VariationalInference`` and ``MonteCarlo`` methods.

Currently, Edward has most of its inference infrastructure within the
``VariationalInference`` class.
The ``MonteCarlo`` class is still under development. We welcome
researchers to make significant advances here!

Let's focus on ``VariationalInference``. The main method in
``VariationalInference`` is ``run()``.

.. code:: python

  class VariationalInference(Inference):
    """Base class for variational inference methods.
    """
    ...
    def run(self, *args, **kwargs):
      """A simple wrapper to run variational inference.
      """
      self.initialize(*args, **kwargs)
      for t in range(self.n_iter+1):
        loss = self.update()
        self.print_progress(t, loss)

      self.finalize()

    ...

First, it calls ``initialize()`` to initialize the algorithm, such as
setting the number of iterations. Then, within a loop it calls
``update()`` which runs one step of inference, as well as
``print_progress()`` for displaying progress; finally, it
calls ``finalize()`` which runs the last steps as the inference
algorithm terminates.

Developing a new variational inference algorithm is as simple as
inheriting from ``VariationalInference`` or one of its derived
classes. ``VariationalInference`` implements many default methods such
as ``run()`` above. Let's go through ``initialize()`` as an example.

.. code:: python

  class VariationalInference(Inference):
    ...
    def initialize(self, ...):
      ...
      if n_minibatch is not None ...
        ...
        slices = tf.train.slice_input_producer(values)
        batches = tf.train.batch(slices, n_minibatch,
                                 num_threads=multiprocessing.cpu_count())
        ...
        self.data = {key: value for key, value in
                     zip(six.iterkeys(self.data), batches)}
      ...
      loss = self.build_loss()
      ...
      optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train = optimizer.minimize(loss, ...)

Three code snippets are highlighted in ``initialize()``: the first
enables batch training with an argument ``n_minibatch`` for the batch
size; the second defines the loss function, building TensorFlow's
computational graph; the third sets up an optimizer to minimize the
loss. These three snippets are applicable to all of variational
inference, and are thus useful defaults for any derived class.

For examples of inference algorithms built in Edward, see the inference
`tutorials <../tutorials>`__.

Variational Models
^^^^^^^^^^^^^^^^^^

A variational model defines a distribution over latent
variables. It is a model of the posterior distribution, specifying
another distribution to approximate it. This is analogous to the way
that probabilistic models specify distributions to approximate the
true data distribution. After inference, the variational model is used
as a proxy to the true posterior.

Edward implements variational models using the ``RandomVariable`` class in
``edward.models``. During inference, each latent variable in the model is binded to a ``RandomVariable`` object, which aims to match the latent variable given data.

All random variable objects, i.e.,
any class inheriting from ``RandomVariable`` in ``edward.models``, takes
as input a shape and optionally, parameter arguments. If left
unspecified, the parameter arguments are trainable parameters during
inference.  The shape denotes the shape of its random variable. For
example:

.. code:: python

  from edward.models import Normal, InvGamma, Beta

  # a vector of 10 random variables
  qz1 = InvGamma(10)

  # a 5 x 2 matrix of random variables
  qz2 = Normal([5, 2])

  # vector of 3 random variables with fixed alpha param
  qz = Beta(3, alpha=tf.ones(3))

Multivariate distributions store their multivariate dimension in the
outer dimension (right-most dimension) of their shape.

.. code:: python

  from edward.models import Dirichlet

  # 1 K-dimensional Dirichlet
  Dirichlet(alpha=np.array([0.1]*K)
  # vector of 5 K-dimensional Dirichlet's
  Dirichlet(alpha=tf.ones([5, K]))

The main methods in each ``RandomVariable`` are ``log_prob()`` and
``sample()``, which mathematically are ``log q(z; \lambda)`` and ``z ~
q(z; \lambda)`` respectively. See their docstrings for more details.

.. works with a list of tensors
.. if there is more than one layer, and a single tensor if only one layer.
.. This arises in the input for ``variational.log_prob(xs)`` as well as the
.. output for ``variational.sample(n)``.

.. There is a nuance worth mentioning why there's a difference in the
.. ``log_prob(xs, zs)`` methods of ``mixture_gaussian.py`` compared to
.. ``mixture_gaussian_map.py``. The former uses three sets of variational
.. distributions; the latter uses one (a point mass). This means the former
.. takes ``zs`` as a list of 3 tensors, and the latter takes ``zs`` as a
.. single tensor. While this isn't satisfactory (the probability model's
.. method should not rely on the variational model downstream), this makes
.. the difference which already currently exists more transparent.

.. explain the ``log_prob()`` nuance for multivariate vs univariate
.. 4distributions.
