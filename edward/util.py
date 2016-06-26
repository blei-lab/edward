import tensorflow as tf
import numpy as np

def cumprod(x):
    """Cumulative product of a tensor along first dimension.
    https://github.com/tensorflow/tensorflow/issues/813

    Parameters
    ----------
    x : tf.Tensor
        vector, matrix, or n-Tensor

    Returns
    -------
    tf.Tensor
        A Tensor with `cumprod` applied along its first dimension.
    """
    values = tf.unpack(x)
    out = []
    prev = tf.ones_like(values[0])
    for val in values:
        s = prev * val
        out.append(s)
        prev = s

    return tf.pack(out)

def digamma(x):
    """Evaluate the digamma function element-wise.

    TensorFlow doesn't have special functions with support for
    automatic differentiation, so use a log/exp/polynomial approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html

    Parameters
    ----------
    x : np.array or tf.Tensor
        scalar, vector, or rank-n tensor

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    twopx = 2.0 + x
    logterm = tf.log(twopx)
    return - (1.0 + 2.0 * x) / (x * (1.0 + x)) - \
           (13.0 + 6.0 * x) / (12.0 * twopx * twopx) + logterm

def dot(x, y):
    """Compute dot product between a Tensor matrix and a Tensor vector.

    Parameters
    ----------
    x : tf.Tensor
        `M x N` matrix or `M` vector (respectively)
    y : tf.Tensor
        `M` vector or `M x N` matrix (respectively)

    Returns
    -------
    tf.Tensor
        `N` vector
    """        
    if len(x.get_shape()) == 1:
        vec = x
        mat = y
        return tf.matmul(tf.expand_dims(vec, 0), mat)
    else:
        mat = x
        vec = y
        return tf.matmul(mat, tf.expand_dims(vec, 1))

def get_dims(x):
    """Get values of each dimension.

    Parameters
    ----------
    x: tf.Tensor
        scalar, vector, matrix, or n-Tensor

    Returns
    -------
    list
        Python list containing dimensions of `x`
    """
    dims = x.get_shape()
    if len(dims) == 0: # scalar
        return [1]
    else: # array
        return [dim.value for dim in dims]

def get_session():
    """Get the globally defined TensorFlow session defined globally.

    If the session is not already defined, then the function will create 
    a global session.

    Returns
    -------
    _ED_SESSION : tf.InteractiveSession
    """
    global _ED_SESSION
    if tf.get_default_session() is None:
        _ED_SESSION = tf.InteractiveSession()

    return _ED_SESSION

def hessian(y, xs):
    """Calculate Hessian of y with respect to each x in xs.

    Parameters
    ----------
    y : tf.Tensor
        Tensor to calculate Hessian of.
    xs : list
        List of TensorFlow variables to calculate with respect to.
        The variables can have different shapes.

    Returns
    -------
    tf.Tensor
        A matrix where each row is

        .. math:: \partial_{xs} ( [ \partial_{xs} y ]_j ).
    """
    # Calculate flattened vector grad_{xs} y.
    grads = tf.gradients(y, xs)
    grads = [tf.reshape(grad, [-1]) for grad in grads]
    grads = tf.concat(0, grads)
    # Loop over each element in the vector.
    mat = []
    d = grads.get_shape()[0]
    for j in range(d):
        # Calculate grad_{xs} ( [ grad_{xs} y ]_j ).
        gradjgrads = tf.gradients(grads[j], xs)
        # Flatten into vector.
        hi = []
        for l in range(len(xs)):
            hij = gradjgrads[l]
            # return 0 if gradient doesn't exist; TensorFlow returns None
            if hij is None:
                hij = tf.zeros(xs[l].get_shape(), dtype=tf.float32)

            hij = tf.reshape(hij, [-1])
            hi.append(hij)

        hi = tf.concat(0, hi)
        mat.append(hi)

    # Form matrix where each row is grad_{xs} ( [ grad_{xs} y ]_j ).
    return tf.pack(mat)

def kl_multivariate_normal(loc_one, scale_one, loc_two=0, scale_two=1):
    """Calculate the KL of multivariate normal distributions with
    diagonal covariances.

    Parameters
    ----------
    loc_one : tf.Tensor
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the mean of a n-dimensional Gaussian
    scale_one : tf.Tensor
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the standard deviation of a n-dimensional Gaussian
    loc_two : tf.Tensor, optional
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the mean of a n-dimensional Gaussian
    scale_two : tf.Tensor, optional
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the standard deviation of a n-dimensional Gaussian

    Returns
    -------
    tf.Tensor
        for scalar or vector inputs, outputs the scalar
        ``KL( N(z; loc_one, scale_one) || N(z; loc_two, scale_two) )``

        for matrix inputs, outputs the vector
        ``[KL( N(z; loc_one[m,:], scale_one[m,:]) || N(z; loc_two[m,:], scale_two[m,:]) )]_{m=1}^M``
    """
    if loc_two == 0 and scale_two == 1:
        return 0.5 * tf.reduce_sum(
            tf.square(scale_one) + tf.square(loc_one) - \
            1.0 - 2.0 * tf.log(scale_one))
    else:
        return 0.5 * tf.reduce_sum(
            tf.square(scale_one/scale_two) + \
            tf.square((loc_two - loc_one)/scale_two) - \
            1.0 + 2.0 * tf.log(scale_two) - 2.0 * tf.log(scale_one), 1)

def lbeta(x):
    """Compute the log of the Beta function, reducing along the last dimension.

    TensorFlow doesn't have special functions with support for
    automatic differentiation, so use a log/exp/polynomial approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html

    Parameters
    ----------
    x : np.array or tf.Tensor
        scalar, vector, matrix, or n-Tensor

    Returns
    -------
    tf.Tensor
        scalar if vector input, rank-(n-1) if rank-n tensor input
    """
    x = tf.cast(tf.squeeze(x), dtype=tf.float32)
    if len(get_dims(x)) == 1:
        return tf.reduce_sum(lgamma(x)) - lgamma(tf.reduce_sum(x))
    else:
        return tf.reduce_sum(lgamma(x), 1) - lgamma(tf.reduce_sum(x, 1))

def lgamma(x):
    """Evaluate the log of the Gamma function element-wise.

    TensorFlow doesn't have special functions with support for
    automatic differentiation, so use a log/exp/polynomial approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html

    Parameters
    ----------
    x : np.array or tf.Tensor
        scalar, vector, matrix, or n-Tensor

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    logterm = tf.log(x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log(xp3)

def log_sum_exp(x):
    """Compute the ``log_sum_exp`` of the elements in x.

    Parameters
    ----------
    x : tf.Tensor
        vector or matrix with second dimension 1

        shape=TensorShape([Dimension(N)])

        shape=TensorShape([Dimension(N), Dimension(1)])

    Returns
    -------
    tf.Tensor
        scalar if vector input, vector if matrix tensor input
    """
    x_max = tf.reduce_max(x)
    return tf.add(x_max, tf.log(tf.reduce_sum(tf.exp(tf.sub(x, x_max)))))

def logit(x):
    """Evaluates :math:`\log(x / (1 - x))` elementwise. 

    Clips all elements to be between :math:`(0,1)`.
    
    Parameters
    ----------
    x : tf.Tensor
        scalar, vector, matrix, or n-Tensor 

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    x = tf.clip_by_value(x, 1e-8, 1.0 - 1e-8)
    return tf.log(x) - tf.log(1.0 - x)

def multivariate_rbf(x, y=0.0, sigma=1.0, l=1.0):
    """Squared-exponential kernel

    .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) \sum_i (x_i - y_i)^2 }

    Parameters
    ----------
    x : tf.Tensor
        scalar, vector, matrix, or n-Tensor 
    y : Optional[tf.Tensor], default 0.0
        scalar, vector, matrix, or n-Tensor 
    sigma : Optional[double], default 1.0
        standard deviation of radial basis function
    l : Optional[double], default 1.0
        lengthscale of radial basis function

    Returns
    -------
    tf.Tensor
        scalar if vector input, rank-(n-1) if n-Tensor input
    """
    return tf.pow(sigma, 2.0) * \
           tf.exp(-1.0/(2.0*tf.pow(l, 2.0)) * \
                  tf.reduce_sum(tf.pow(x - y , 2.0)))

def rbf(x, y=0.0, sigma=1.0, l=1.0):
    """Squared-exponential kernel element-wise

    .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) (x_i - y_i)^2 }

    Parameters
    ----------
    x : tf.Tensor
        scalar, vector, matrix, or n-Tensor 
    y : Optional[tf.Tensor], default 0.0
        scalar, vector, matrix, or n-Tensor 
    sigma : Optional[double], default 1.0
        standard deviation of radial basis function
    l : Optional[double], default 1.0
        lengthscale of radial basis function

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    return tf.pow(sigma, 2.0) * \
           tf.exp(-1.0/(2.0*tf.pow(l, 2.0)) * tf.pow(x - y , 2.0))

def set_seed(x):
    """Set seed for both NumPy and TensorFlow.

    Parameters
    ----------
    x : double
        seed
    """
    np.random.seed(x)
    tf.set_random_seed(x)

def softplus(x):
    """Elementwise Softplus function

    .. math:: \log(1 + \exp(x))

    TensorFlow can't currently autodiff through tf.nn.softplus().

    Parameters
    ----------
    x : tf.Tensor
        scalar, vector, matrix, or n-Tensor 

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    return tf.log(1.0 + tf.exp(x))

class VarStoreMethod(object):
    """Convenience base class for registered methods that create variables.

    This tracks the variables and requries subclasses to provide a ``__call__``
    method.

    This is taken from PrettyTensor.
    https://github.com/google/prettytensor/blob/
    c9b69fade055d0eb35474fd23d07c43c892627bc/prettytensor/
    pretty_tensor_class.py#L1497
    """

    def __init__(self):
        self.vars = {}

    def variable(self, var_name, shape, init=tf.random_normal_initializer(), 
                 dt=tf.float32, train=True):
        """Adds a named variable to this bookkeeper or returns an existing one.
        Variables marked train are returned by the training_variables method. If
        the requested name already exists and it is compatible (same shape, dt
        and train) then it is returned. In case of an incompatible type, an
        exception is thrown.

        Parameters
        ----------
        var_name : string
            The unique name of this variable.  If a variable with the same
            name exists, then it is returned.
        shape : tf.TensorShape
            The shape of the variable.
        init : 
            The init function to use or a Tensor to copy.

            Defaults to ``tf.random_normal_initializer()``.
        dt : 
            The datatype, defaults to ``tf.float32``.  This will automatically 
            extract the base ``dtype``.
        train : bool
            Whether or not the variable should be trained.

        Returns
        -------
        v : string
          The input `var_name`

        Raises
        ------
        ValueError: 
            if reuse is ``False`` (or unspecified and allow_reuse is ``False``)
            and the variable already exists or if the specification of a reused
            variable does not match the original.
        """

        # Make sure it is a TF dtype and convert it into a base dtype.
        dt = tf.as_dtype(dt).base_dtype
        if var_name in self.vars:
            v = self.vars[var_name]
            if v.get_shape() != shape:
                raise ValueError(
                    'Shape mismatch: %s vs %s. Perhaps a UnboundVariable had '
                    'incompatible values within a graph.' % (v.get_shape(), shape))
            return v
        elif callable(init):

            v = tf.get_variable(var_name,
                              shape=shape,
                              dtype=dt,
                              initializer=init,
                              trainable=train)
            self.vars[var_name] = v
            return v
        else:
            v = tf.convert_to_tensor(init, name=var_name, dtype=dt)
            v.get_shape().assert_is_compatible_with(shape)
            self.vars[var_name] = v
            return v

class VARIABLE(VarStoreMethod):
    """A simple wrapper to contain variables. It will create a TensorFlow
    variable the first time it is called and return the variable; in
    subsequent calls, it will simply return the variable and not
    create the TensorFlow variable again.

    This enables variables to be stored outside of classes which
    depend on parameters. It is a useful application for parametric
    distributions whose parameters may or may not be random (e.g.,
    through a prior), and for inverse mappings such as auto-encoders
    where we'd like to store inverse mapping parameters outside of the
    distribution class.
    """
    def __call__(self, name, shape):
        self.name = name
        return self.variable(name, shape)

Variable = VARIABLE()
