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

For examples of variational models built in Edward, see the end-to-end
`tutorials <../tutorials.html>`__ which use variational inference.
