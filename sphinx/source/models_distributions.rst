Distributions
^^^^^^^^^^^^^

All distribution objects, i.e., any class inheriting from
``Distribution`` in ``edward.models``, now takes as input a shape and
optionally, parameter arguments. The shape denotes the shape of its
random variable. For example:

.. code:: python

    from edward.models import Normal

    # 5 x 2 matrix of random variables
    x = Normal([5, 2])
    x.sample(size=10)

The output is a set of samples with shape ``(10, 5, 2)``, where size is
the outer dimension.

This generalizes the previously cryptic ``num_factors`` argument and
enables tensors of random variables. Note also that as a result, all
sample methods—whether in be in the container object ``Variational()``,
distribution objects ``Distribution()``, or ``edward.stats`` sampling
``rvs()``—will draw samples where the size is the outer dimension. For
example:

.. code:: python

    from edward.stats import bernoulli

    x = bernoulli.rvs(p=0.5, size=1)
    print(x.shape)
    ## (1,)
    x = bernoulli.rvs(p=np.array([0.5]), size=1)
    print(x.shape)
    ## (1, 1)
    x = bernoulli.rvs(p=np.array([0.5, 0.2]), size=3)
    print(x.shape)
    ## (3, 2)

This goes against SciPy behavior (which is inconsistent anyways):
``size=1`` typically results in the same output dimension as the
parameter rather than ``size x shape(parameter)``. Our standard adheres
to the behavior in the ``tf.contrib.distributions`` library.

The container object ``Variational()`` now works with a list of tensors
if there is more than one layer, and a single tensor if only one layer.
This arises in the input for ``variational.log_prob(xs)`` as well as the
output for ``variational.sample(size)``. Carrying around a list of
tensors makes packing and unpacking of latent variables much easier to
work with; see the mixture of Gaussians example. Carrying around a
tensor if only one layer is convenient for probability models that deal
with only one set of latent variables; it's easier to work directly with
``zs`` rather than do ``zs[0]``; see the Beta-Bernoulli or hierarchical
linear model examples.

There is a nuance worth mentioning why there's a difference in the
``log_prob(xs, zs)`` methods of ``mixture_gaussian.py`` compared to
``mixture_gaussian_map.py``. The former uses three sets of variational
distributions; the latter uses one (a point mass). This means the former
takes ``zs`` as a list of 3 tensors, and the latter takes ``zs`` as a
single tensor. While this isn't satisfactory (the probability model's
method should not rely on the variational model downstream), this makes
the difference which already currently exists more transparent.
