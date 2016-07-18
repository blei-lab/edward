Inference
---------

An inference algorithm infers the posterior for a particular model
``p(x, z)`` and data set ``x``. It is the distribution of the latent
variables given data, ``p(z | x)``. For more details, see the
`Inference of Probability Models tutorial <../tut_inference.html>`__.

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
      def __init__(self, model, data=None):
          ...

It takes as input a probabilistic model ``model`` and dataset
``data``.
For more details, see the
`Model API <models.html>`__
and
`Data API <data.html>`__.

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
      def __init__(self, model, data=None):
          super(MonteCarlo, self).__init__(model, data)

      ...


  class VariationalInference(Inference):
      """Base class for variational inference methods.
      """
      def __init__(self, model, variational, data=None):
          """Initialization.
          ...
          """
          super(VariationalInference, self).__init__(model, data)
          self.variational = variational

      ...

Hybrid methods and novel paradigms outside of ``VariationalInference``
and ``MonteCarlo`` are also possible in Edward. For example, one can
write a class derived from ``Inference`` directly, or inherit to
carry both ``VariationalInference`` and ``MonteCarlo`` methods.

Currently, Edward has most of its inference infrastructure within the
``VariationalInference`` class.
The ``MonteCarlo`` class is still under development. We welcome
researchers to make significant advances here!

Let's focus on ``VariationalInference``. In addition to a model and
data as input, ``VariationalInference`` takes in a variational
model ``variational``, which serves as a model of the posterior
distribution. For more details, see the Variational Models section
below.

The main method in ``VariationalInference`` is ``run()``.

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
`tutorials <../tutorials.html>`__.

Variational Models
^^^^^^^^^^^^^^^^^^

A variational model defines a distribution over latent
variables. It is a model of the posterior distribution, specifying
another distribution to approximate it. This is analogous to the way
that probabilistic models specify distributions to approximate the
true data distribution. After inference, the variational model is used
as a proxy to the true posterior.

Edward implements variational models using the ``Variational`` class in
``edward.models``. For example, the following instantiates an empty
container for the variational distribution.

.. code:: python

    from edward.models import Variational

    variational = Variational()

To add distributions to this object, use the ``add()`` method, which
is used to add ``RandomVariable`` objects.  All random variable objects, i.e.,
any class inheriting from ``RandomVariable`` in ``edward.models``, takes
as input a shape and optionally, parameter arguments. If left
unspecified, the parameter arguments are trainable parameters during
inference.  The shape denotes the shape of its random variable. For
example:

.. code:: python

    from edward.models import Variational, Normal, Beta

    # first, add a vector of 10 random variables
    # second, add a 5 x 2 matrix of random variables
    variational = Variational()
    variational.add(InvGamma(10))
    variational.add(Normal([5, 2]))

    # vector of 3 random variables with fixed alpha param
    variational = Variational()
    variational.add(Beta(3, alpha=tf.ones(3)))

Multivariate distributions store their multivariate dimension in the
outer dimension (right-most dimension) of their shape.

.. code:: python

    from edward.models import Dirichlet

    # 1 K-dimensional Dirichlet
    Dirichlet(alpha=np.array([0.1]*K)
    # vector of 5 K-dimensional Dirichlet's
    Dirichlet(alpha=tf.ones([5, K]))

The main methods in ``Variational`` are ``log_prob()`` and
``sample()``, which mathematically are ``log q(z; \lambda)`` and ``z ~
q(z; \lambda)`` respectively.

``samples(n)`` takes as input the number of samples and returns a list
of TensorFlow tensors, each of whose shape is ``(n, ) + self.shape`` for
each random variable object within the container. ``log_prob(xs)`` takes
as input a list of TensorFlow tensors, and returns a vector of density
evaluations, one for each sample ``x`` in ``xs``.

The ordering of the addition to the container matters. This defines
the ordering of the lists for the output of ``sample()`` and the input
of ``log_prob()``.
(As an example, see the `mixture of Gaussians
<https://github.com/blei-lab/edward/blob/master/examples/mixture_gaussian.py>`__.)

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
