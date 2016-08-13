from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


class DelayedTensor(object):
    """Node in a metagraph, representing a random variable in a model.

    The metagraph carries instructions about how to build model
    tensors in the computational graph. Graph construction is delayed
    until one calls `build()`.

    Attributes
    ----------
    conditioning_set : list
        Default inputs to stochastic tensor when building.
    lambda_fn : function
        Function of conditioning set, returning a stochastic tensor.

    Examples
    --------
    >>> mu = tf.constant([0.0])
    >>> sigma = tf.constant([1.0])
    >>> x = DelayedTensor([mu, sigma],
    ...   lambda cond_set: sg.DistributionTensor(distributions.Normal,
    ...                                          mu=cond_set[0],
    ...                                          sigma=cond_set[1]))
    >>> x_tensor = x.build()
    >>>
    >>> x = ed.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> y = ed.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> out = ed.matmul(x, y)
    >>> out_tensor = out.build()
    >>>
    >>> sess = tf.Session()
    >>> sess.run(out_tensor)

    `x.build()` builds the distribution tensor, defaulting to the
    initialized conditioning set. `sess.run(x_tensor)` returns samples
    from the generative process.
    """
    def __init__(self, conditioning_set, lambda_fn):
        self.conditioning_set = conditioning_set
        self.lambda_fn = lambda_fn

    def build(self, conditioning_set=None, built_dict=None, latent_vars=None):
        """Build tensor according to conditioning set.

        Parameters
        ----------
        conditioning_set : list, optional
            Conditioning set of the stochastic tensor. Default is to
            build it according to `self.conditioning_set`. Any
            elements that are `None` in a passed-in `conditioning_set`
            default to the corresponding element in
            `self.conditioning_set`.
        built_dict : dict, optional
            Dictionary of `DelayedTensor`s binded to their built
            stochastic tensor. Will use any built tensors from random
            variables in this dictionary that `self` depends on.
            `built_dict` is also modified in-place to include any
            random variables built during this function.
        latent_vars : dict, optional
            Dictionary of `DelayedTensor`s binded to a
            value. For a `DelayedTensor` `x` in
            the conditioning set, we will condition on
            `latent_vars[x]` instead. For example, this is
            used to replace conditioning on the prior with
            conditioning on the posterior (without explicitly passing
            in `conditioning_set` to do so).

        Returns
        -------
        tf.Tensor
            Stochastic tensor.
        """
        if built_dict is None:
            built_dict = {}

        # Do nothing if tensor is already built in `built_dict`.
        if self in built_dict:
            return built_dict[self]

        # Default to the initialized conditioning set.
        if conditioning_set is None:
            conditioning_set = self.conditioning_set

        for i, x in enumerate(conditioning_set):
            # Set any None values to its corresponding default.
            if x is None:
                x = self.conditioning_set[i]

            if isinstance(x, DelayedTensor):
                # Set to corresponding value in `latent_vars`
                # if it is available.
                if latent_vars is not None:
                    if x in latent_vars:
                        x = latent_vars[x]

                # Use and store built stochastic tensors in
                # `built_dict` if it is available.
                if x in built_dict:
                    x_tensor = built_dict[x]
                else:
                    # Recursively build any DelayedTensor's in
                    # the conditioning set.
                    x_tensor = x.build(built_dict=built_dict,
                                       latent_vars=latent_vars)
            else:
                x_tensor = x

            conditioning_set[i] = x_tensor

        rv_tensor = self.lambda_fn(conditioning_set)
        built_dict[self] = rv_tensor
        return rv_tensor


class DelayedOperation(DelayedTensor):
    """Wrapper for delayed tensor using a pre-existing TensorFlow
    operation."""
    def __init__(self, op, *args, **kwargs):
        cond_set = list(args) + list(six.itervalues(kwargs))
        self.op = op
        self.args_len = len(args)
        self.kwargs_keys = list(six.iterkeys(kwargs))
        def lambda_fn(cond_set):
            args = cond_set[:self.args_len]
            kwargs_values = cond_set[self.args_len:]
            kwargs = {key: value for key, value in zip(self.kwargs_keys, kwargs_values)}
            return self.op(*args, **kwargs)

        super(DelayedOperation, self).__init__(cond_set, lambda_fn)


class Bernoulli(DelayedOperation):
    """
    Examples
    --------
    >>> p = tf.constant([0.5])
    >>> x = Bernoulli(p=p)
    >>>
    >>> z1 = tf.constant([[2.0, 8.0]])
    >>> z2 = tf.constant([[1.0, 2.0]])
    >>> x = Bernoulli(p=ed.matmul(z1, z2))
    """
    def __init__(self, *args, **kwargs):
        super(Bernoulli, self).__init__(sg.DistributionTensor, distributions.Bernoulli, *args, **kwargs)


class Beta(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Beta, self).__init__(sg.DistributionTensor, distributions.Beta, *args, **kwargs)


class Categorical(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Categorical, self).__init__(sg.DistributionTensor, distributions.Categorical, *args, **kwargs)


class Chi2(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Chi2, self).__init__(sg.DistributionTensor, distributions.Chi2, *args, **kwargs)


class Dirichlet(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Dirichlet, self).__init__(sg.DistributionTensor, distributions.Dirichlet, *args, **kwargs)


class DirichletMultinomial(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DirichletMultinomial, self).__init__(sg.DistributionTensor, distributions.DirichletMultinomial, *args, **kwargs)


class Exponential(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Exponential, self).__init__(sg.DistributionTensor, distributions.Exponential, *args, **kwargs)


class Gamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Gamma, self).__init__(sg.DistributionTensor, distributions.Gamma, *args, **kwargs)


class InverseGamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(InverseGamma, self).__init__(sg.DistributionTensor, distributions.InverseGamma, *args, **kwargs)


class Laplace(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Laplace, self).__init__(sg.DistributionTensor, distributions.Laplace, *args, **kwargs)


class MultivariateNormalCholesky(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultivariateNormalCholesky, self).__init__(sg.DistributionTensor, distributions.MultivariateNormalCholesky, *args, **kwargs)


class MultivariateNormalDiag(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultivariateNormalDiag, self).__init__(sg.DistributionTensor, distributions.MultivariateNormalDiag, *args, **kwargs)


class MultivariateNormalFull(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultivariateNormalFull, self).__init__(sg.DistributionTensor, distributions.MultivariateNormalFull, *args, **kwargs)


class Normal(DelayedOperation):
    """
    Examples
    --------
    >>> mu = Normal(mu=tf.constant(0.0), sigma=tf.constant(1.0)])
    >>> x = Normal(mu=mu, sigma=tf.constant([1.0]))
    """
    def __init__(self, *args, **kwargs):
        super(Normal, self).__init__(sg.DistributionTensor, distributions.Normal, *args, **kwargs)


class StudentT(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(StudentT, self).__init__(sg.DistributionTensor, distributions.StudentT, *args, **kwargs)


class TransformedDistribution(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TransformedDistribution, self).__init__(sg.DistributionTensor, distributions.TransformedDistribution, *args, **kwargs)


class Uniform(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Uniform, self).__init__(sg.DistributionTensor, distributions.Uniform, *args, **kwargs)
