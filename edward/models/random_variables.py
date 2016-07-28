from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


class RandomVariable(object):
    """Node in a metagraph, which defines a model.

    The metagraph carries instructions about how to build model
    tensors in the computational graph. Graph construction is delayed
    until one calls `build()`.

    Attributes
    ----------
    lambda_fn : function
        Function of conditioning set, returning a stochastic tensor.
    conditioning_set : list
        Default inputs to stochastic tensor when building.

    Examples
    --------
    >>> mu = tf.constant([0.0])
    >>> sigma = tf.constant([1.0])
    >>> x = RandomVariable(
    ...   lambda mu, sigma: sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma),
    ...   [mu, sigma])
    >>> x_tensor = x.build()

    `x.build()` builds the distribution tensor, defaulting to the
    initialized conditioning set. `sess.run(x_tensor)` returns samples
    from the generative process.
    """
    def __init__(self, lambda_fn, conditioning_set):
        self.lambda_fn = lambda_fn
        self.conditioning_set = conditioning_set

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
            Dictionary of `RandomVariable`s binded to their built
            stochastic tensor. Will use any built tensors from random
            variables in this dictionary that `self` depends on.
            `built_dict` is also modified in-place to include any
            random variables built during this function.
        latent_vars : dict, optional
            Dictionary of `RandomVariable`s binded to a
            value. For a `RandomVariable` `x` in
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

            if isinstance(x, RandomVariable):
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
                    # Recursively build any RandomVariable's in
                    # the conditioning set.
                    x_tensor = x.build(built_dict=built_dict)
            else:
                x_tensor = x

            conditioning_set[i] = x_tensor

        rv_tensor = self.lambda_fn(conditioning_set)
        built_dict[self] = rv_tensor
        return rv_tensor


class Bernoulli(RandomVariable):
    """
    Examples
    --------
    >>> p = tf.constant([0.5])
    >>> x = Bernoulli([p])
    >>>
    >>> z1 = tf.constant([2.0, 8.0])
    >>> z2 = tf.constant([1.0, 2.0])
    >>> x = Bernoulli([z1, z2], lambda cond_set: tf.matmul(cond_set[0], cond_set[1]))
    """
    def __init__(self, cond_set, p_lambda_fn=None):
        # If lambda functions are not passed in, default is to assume
        # `cond_set` is passed in as [p].
        if p_lambda_fn is None:
            p_lambda_fn = lambda cond_set: cond_set[0]

        lambda_fn = lambda cond_set: \
            sg.DistributionTensor(distributions.Bernoulli, p_lambda_fn(cond_set))
        super(Bernoulli, self).__init__(lambda_fn, cond_set)


class Normal(RandomVariable):
    """
    Examples
    --------
    >>> mu = Normal([tf.constant(0.0), tf.constant(1.0)])
    >>> sigma = tf.constant([1.0])
    >>> x = Normal([mu, sigma])
    """
    # TODO make single lambda fn or both
    def __init__(self, cond_set, mu_lambda_fn=None, sigma_lambda_fn=None):
        # If lambda functions are not passed in, default is to assume
        # `cond_set` is passed in as [mu, sigma].
        if mu_lambda_fn is None:
            mu_lambda_fn = lambda cond_set: cond_set[0]

        if sigma_lambda_fn is None:
            sigma_lambda_fn = lambda cond_set: cond_set[1]

        lambda_fn = lambda cond_set: \
            sg.DistributionTensor(distributions.Normal,
                                  mu_lambda_fn(cond_set), sigma_lambda_fn(cond_set))
        super(Normal, self).__init__(lambda_fn, cond_set)
