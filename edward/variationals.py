from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli, beta, norm, dirichlet, invgamma
from edward.util import Variable

class Variational:
    """
    A stack of variational families and their parameterizations.
    It is represented as a directed acyclic graph, where a node is a
    variational family or parameterization. An edge i -> j exists if
    node i is a prior on the parameters of node j or it is a
    parameterization of node j.

    Parameters
    ----------
    graph : list
        List of lists where the ith list is the ith layer, and each
        element in the list (node in a layer) is a variational family
        or mapping. The first layer can contain only variational
        families.

    Notes
    -----
    Currently, nodes are only connected to the node directly above it.
    """
    def __init__(self, graph=[[]]):
        self.graph = graph
        self.num_vars = 0 # num of posterior latent variables
        self.num_params = 0 # num of variational parameters
        self.is_reparam = True # if the variational family is reparameterizable

    def add(self, node):
        """
        Add a node instance horizontally.

        Parameters
        ----------
        node : variational family or parameterization
        """
        self.graph[-1] += [node]
        self.is_reparam = self.is_reparam and 'reparam' in node.__class__.__dict__

        if len(self.graph) == 1:
            if not isinstance(node, Likelihood):
                raise

            self.num_params += node.num_params
            self.num_vars += node.num_vars
        else:
            connected_node = self.graph[-2][self.horizontal_index]
            if hasattr(node, '__call__'):
                mapping = Mapping(node, connected_node.num_params)
                connected_node.mapping = mapping
                self.num_params += mapping.num_params
            elif isinstance(node, Likelihood):
                # For now this does nothing
                raise
            else:
                raise

            self.num_params -= connected_node.num_params
            self.horizontal_index += 1

    def layer(self):
        """Declare a new layer. Adding now adds to this layer."""
        self.graph += [[]]
        self.horizontal_index = 0 # to keep track when adding to a layer

    def mapping(self, x):
        return [node.mapping(x) for node in self.graph[0]]

    # The following methods all deal with only the first layer.
    def set_params(self, params):
        [node.set_params(params[i]) for i,node in enumerate(self.graph[0])]

    def print_params(self, sess):
        [node.print_params(sess) for node in self.graph[0]]

    def sample_noise(self, size):
        eps_list = [node.sample_noise((size[0], node.num_vars))
                    for node in self.graph[0]]
        return np.concatenate(eps_list, axis=1)

    def reparam(self, eps):
        z_list = []
        start = final = 0
        for node in self.graph[0]:
            final += node.num_vars
            z_list += [node.reparam(eps[:, start:final])]
            start = final

        return tf.concat(1, z_list)

    def sample(self, size, sess):
        #z_list = [node.sample((size[0], node.num_vars), sess)
        #          for node in self.graph[0]]
        # This is temporary to deal with reparameterizable ones.
        z_list = []
        for node in self.graph[0]:
            z_node = node.sample((size[0], node.num_vars), sess)
            if isinstance(node, Normal):
                z_node = sess.run(z_node)

            z_list += [z_node]

        return np.concatenate(z_list, axis=1)

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda)"""
        start = final = 0
        for node in self.graph[0]:
            final += node.num_vars
            if start + i < final:
                return node.log_prob_zi(i, z[:, start:final])

            i = i - node.num_vars
            start = final

        raise IndexError()

class Mapping:
    def __init__(self, f, output_dim):
        self.f = f
        self.output_dim = output_dim

        self.num_params = 0 # TODO figure this out from f

    def __call__(self, x):
        return self.f(x, output_dim=self.output_dim)

    def set_params(self):
        # This will be used if placing a prior over its parameters.
        pass

# TODO possibly rename to variational family or maybe distribution if
# it also generalizes to work for probability models
class Likelihood:
    """
    Base class for variational likelihoods, q(z | lambda).

    Parameters
    ----------
    num_vars : int
        Number of latent variables.
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.num_params = None # number of local variational parameters
        # TODO attribute for number of global variational parameters

    def mapping(self, x):
        """
        A mapping from data point x -> lambda, the local variational
        parameters, which are parameters specific to x.

        Parameters
        ----------
        x : Data
            Data point

        Returns
        -------
        list
            A list of TensorFlow tensors, where each element is a
            particular set of local parameters.

        Notes
        -----
        In classical variational inference, the mapping can be
        interpreted as the collection of all local variational
        parameters; the output is simply the projection to the
        relevant subset of local parameters.

        For local variational parameters with constrained support, the
        mapping additionally acts as a transformation. The parameters
        to be optimized live on the unconstrained space; the output of
        the mapping is then constrained variational parameters.

        Global parameterizations are useful to prevent the parameters
        of this mapping to grow with the number of data points, and
        also as an implicit regularization. This is known as inverse
        mappings in Helmholtz machines and variational auto-encoders,
        and parameter tying in message passing. The mapping is a
        function of data point with a fixed number of parameters, and
        it tries to (in some sense) "predict" the best local
        variational parameters given this lower rank.
        """
        raise NotImplementedError()

    def set_params(self, params):
        """
        This sets the parameters of the variational family, for use in
        other methods of the class.

        Parameters
        ----------
        params : list
            Each element in the list is a particular set of local parameters.
        """
        raise NotImplementedError()

    # TODO use __str__(self):
    def print_params(self, sess):
        raise NotImplementedError()

    def sample_noise(self, size):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)

        Returns
        -------
        np.ndarray
            n_minibatch x dim(lambda) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.
        """
        raise NotImplementedError()

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        raise NotImplementedError()

    def sample(self, size, sess=None):
        """
        z ~ q(z | lambda)

        Parameters
        ----------
        sess : tf.Session, optional

        Returns
        -------
        np.ndarray
            n_minibatch x dim(z) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.

        The method defaults to sampling noise and reparameterizing it
        (which will raise an error if this is not possible).
        """
        return self.reparam(self.sample_noise(size))

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        raise NotImplementedError()

class Bernoulli(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Bernoulli(z[i] | lambda[i])
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        if self.num_vars == 1:
            self.num_params = self.num_vars
        else:
            self.num_params = self.num_vars - 1

        self.p = None

    def mapping(self, x):
        p = Variable("p", [self.num_params])
        # Constrain parameters to lie on simplex.
        p_const = tf.sigmoid(p)
        if self.num_vars > 1:
            p_const = tf.concat(0,
                [p_const, tf.expand_dims(1.0 - tf.reduce_sum(p_const), 0)])

        return [p_const]

    def set_params(self, params):
        self.p = params[0]

    def print_params(self, sess):
        p = sess.run(self.p)
        print("probability:")
        print(p)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        p = sess.run(self.p)
        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = bernoulli.rvs(p[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        return bernoulli.logpmf(z[:, i], self.p[i])

class Beta(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Beta(z[i] | lambda[i])
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_params = 2*self.num_vars
        self.a = None
        self.b = None

    def mapping(self, x):
        alpha = Variable("alpha", [self.num_vars])
        beta = Variable("beta", [self.num_vars])
        return [tf.nn.softplus(alpha), tf.nn.softplus(beta)]

    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]

    def print_params(self, sess):
        a, b = sess.run([self.a, self.b])
        print("shape:")
        print(a)
        print("scale:")
        print(b)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([self.a, self.b])
        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = beta.rvs(a[d], b[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        return beta.logpdf(z[:, i], self.a[i], self.b[i])

class Dirichlet(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Dirichlet(z[i] | lambda[i])
    (each z[i] here is K-dimensional)
    """
    def __init__(self, num_pis, K):
        self.num_pis = num_pis # number of probability vectors
        Likelihood.__init__(self, num_pis*K)
        self.num_params = K * num_pis
        self.K = K
        self.alpha = None

    def mapping(self, x):
        alpha = Variable("dirichlet_alpha", [self.num_pis, self.K])
        return [tf.nn.softplus(alpha)]

    def set_params(self, params):
        self.alpha = params[0]

    def print_params(self, sess):
        alpha = sess.run(self.alpha)
        print("concentration vector:")
        print(alpha)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        alpha = sess.run(self.alpha)
        z = np.zeros(size)
        for pi in xrange(self.num_pis):
            z[:, (pi*self.K):((pi+1)*self.K)] = dirichlet.rvs(alpha[pi, :], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        # a hack for now
        if i >= self.num_vars:
            raise

        if i == 0:
            # TODO take logpdf of just one of the probability vectors
            return dirichlet.logpdf(z[:, :], self.alpha[0, :])

        if i >= 1:
            return tf.constant(0.0, dtype=tf.float32)

class Normal(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Normal(z[i] | lambda[i])
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_params = 2*self.num_vars
        self.m = None
        self.s = None

    def mapping(self, x):
        mean = Variable("mu", [self.num_vars])
        stddev = Variable("sigma", [self.num_vars])
        return [tf.identity(mean), tf.nn.softplus(stddev)]

    def set_params(self, params):
        self.m = params[0]
        self.s = params[1]

    def print_params(self, sess):
        m, s = sess.run([self.m, self.s])
        print("mean:")
        print(m)
        print("std dev:")
        print(s)

    def sample_noise(self, size):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        # Not using this, since TensorFlow has a large overhead
        # whenever calling sess.run().
        #samples = sess.run(tf.random_normal(self.samples.get_shape()))
        return norm.rvs(size=size)

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        return self.m + eps * self.s

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        mi = self.m[i]
        si = self.s[i]
        return tf.pack([norm.logpdf(zm[i], mi, si)
                        for zm in tf.unpack(z)])

    # TODO entropy is bugged
    #def entropy(self):
    #    return norm.entropy(self.transform_s(self.s_unconst))

class InvGamma(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Inv_Gamma(z[i] | lambda[i])
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_params = 2*self.num_vars
        self.a = None
        self.b = None

    def mapping(self, x):
        alpha = Variable("alpha", [self.num_vars])
        beta = Variable("beta", [self.num_vars])
        return [tf.nn.softplus(alpha), tf.nn.softplus(beta)]

    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]

    def print_params(self, sess):
        a, b = sess.run([self.a, self.b])
        print("shape:")
        print(a)
        print("scale:")
        print(b)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([self.a, self.b])
        z = np.zeros(size)
        for d in range(self.num_vars):
            z[:, d] = invgamma.rvs(a[d], b[d], size=size[0])

        return z

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i >= self.num_vars:
            raise

        return invgamma.logpdf(z[:, i], self.a[i], self.b[i])

class PointMass(Likelihood):
    """
    Point mass variational family
    """
    def __init__(self, num_vars, transform=tf.identity):
        Likelihood.__init__(self, num_vars)
        self.num_params = self.num_vars
        self.transform = transform
        self.params = None

    def mapping(self, x):
        params = Variable("params", [self.num_vars])
        return [self.transform(params)]

    def set_params(self, params):
        self.params = params[0]

    def print_params(self, sess):
        params = sess.run(self.params)
        print("parameter values:")
        print(params)

    def get_params(self):
        # Return a matrix to be compatible with probability model
        # methods which assume the input is possibly a mini-batch of
        # parameter samples (used for black box variational methods).
        return tf.reshape(self.params, [1, self.num_params])
