Inference
---------

An inference algorithm infers the posterior for a particular model
``p(x, z)`` and data set ``x``. It is the distribution of the latent
variables given data, ``p(z | x)``. For more details, see the
`Inference of Probability Models tutorial <../tut_inference.html>`__.

Edward uses abstract base classes and class inheritance to provide a
hierarchy of inference methods, all of which are easily extensible.
This enables fast experimentation and research on top of existing
inference methods, whether it be developing new black box inference
algorithms or developing new model-specific inference algorithms which
are tailored to a particular model or restricted class of models.
We detail this below.

.. image:: ../images/inference_structure.png

*Dependency graph of inference methods. Nodes are classes in Edward
and arrows represent class inheritance.*

There is a abstract base class ``Inference``, from which all inference
methods are derived from.

.. code:: python

  class Inference(object):
      """Base class for Edward inference methods.
      ...
      """
      def __init__(self, model, data=None):
          ...

It takes as input a probabilistic model ``model`` and dataset
``data``.
For more details, see the
`Probabilistic Models API <models.html>`__
and
`Data API <data.html>`__.

We categorize inference under two paradigms:
``VariationalInference`` and ``MonteCarlo`` (or more plainly,
optimization and sampling). These inherit from ``Inference`` and each
have their own default methods.

.. code:: python

  class MonteCarlo(Inference):
      """Base class for Monte Carlo inference methods.
      """
      def __init__(self, *args, **kwargs):
          super(MonteCarlo, self).__init__(*args, **kwargs)

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

Currently, Edward has most of its inference infrastructure within the
``VariationalInference`` class.
The ``MonteCarlo`` class is still under development. We welcome
contributors to make significant advances here!

Let's focus on ``VariationalInference``. In addition to a model and
data as input, ``VariationalInference`` also takes in a variational
model ``variational``, which serves as a model of the posterior
distribution. For more details, see the Variational Models API below.

The main method in ``VariationalInference`` is ``run()``.

.. code:: python

  class VariationalInference(Inference):
      """Base class for variational inference methods.
      """
      ...
      def run(self, *args, **kwargs):
          """A simple wrapper to run variational inference.
          ...
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
``print_progress()`` for possibly displaying diagnostics; finally, it
calls ``finalize()`` which runs the final steps as the inference
algorithm terminates.

Developing a new variational inference algorithm is as simple as
inheriting from ``VariationalInference`` or one of its derived
classes. ``VariationalInference`` implements many default methods such
as ``run()`` above. For example, ``initialize()`` creates a TensorFlow
optimizer and builds the computational graph for running the
algorithm. It calls the method ``build_loss()``, which returns a node
to differentiate for gradient-based optimization.  ``build_loss()`` is
not implemented in ``VariationalInference`` and must be defined in a
derived class defining a variational inference algorithm. As another
example, ``update()`` runs a TensorFlow session to run one step of the
optimizer. It also fetches ``self.loss`` which is a node in the
computational graph, forming the objective value given the current
state of the graph. This field must also be implemented in a derived
class.

Nothing in ``Inference`` says anything about the class of models that
an inference algorithm must work with. Thus one can build inference
algorithms which are tailored to a smaller class of models than the
general class available in Edward, or even tailor it to a single model.

Hybrid methods and novel paradigms outside of ``VariationalInference``
and ``MonteCarlo`` are also possible in Edward. For example, one can
write a class derived from ``Inference`` directly, or inherited to
carry both ``VariationalInference`` and ``MonteCarlo`` methods.

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
is used to add Distribution objects.  All distribution objects, i.e.,
any class inheriting from ``Distribution`` in ``edward.models``, takes
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
each distribution object within the container. ``log_prob(xs)`` takes
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
