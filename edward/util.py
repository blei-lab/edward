from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

def cumprod(xs):
    """Cumulative product of a tensor along first dimension.

    https://github.com/tensorflow/tensorflow/issues/813

    Parameters
    ----------
    xs : tf.Tensor
        vector, matrix, or n-Tensor

    Returns
    -------
    tf.Tensor
        A Tensor with `cumprod` applied along its first dimension.

    Raises
    ------
    InvalidArgumentError
        If the input has Inf or NaN values.
    """
    dependencies = [tf.verify_tensor_all_finite(xs, msg='')]
    xs = control_flow_ops.with_dependencies(dependencies, xs)

    values = tf.unpack(xs)
    out = []
    prev = tf.ones_like(values[0])
    for val in values:
        s = prev * val
        out.append(s)
        prev = s

    result = tf.pack(out)
    return result


def dot(x, y):
    """Compute dot product between a Tensor matrix and a Tensor vector.

    If x is a ``[M x N]`` matrix, then y is a ``M``-vector.

    If x is a ``M``-vector, then y is a ``[M x N]`` matrix.

    Parameters
    ----------
    x : tf.Tensor
        ``M x N`` matrix or ``M`` vector (see above)
    y : tf.Tensor
        ``M`` vector or ``M x N`` matrix (see above)

    Returns
    -------
    tf.Tensor
        ``N``-vector

    Raises
    ------
    InvalidArgumentError
        If the inputs have Inf or NaN values.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                    tf.verify_tensor_all_finite(y, msg='')]
    x = control_flow_ops.with_dependencies(dependencies, x)
    y = control_flow_ops.with_dependencies(dependencies, y)

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
    x : tf.Tensor or np.ndarray
        scalar, vector, matrix, or n-tensor

    Returns
    -------
    list
        Python list containing dimensions of `x`
    """
    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        dims = x.get_shape()
        if len(dims) == 0: # scalar
            return []
        else: # array
            return [dim.value for dim in dims]
    elif isinstance(x, np.ndarray):
        return list(x.shape)
    else:
        raise NotImplementedError()


def get_session():
    """Get the globally defined TensorFlow session.

    If the session is not already defined, then the function will create
    a global session.

    Returns
    -------
    _ED_SESSION : tf.InteractiveSession
    """
    global _ED_SESSION
    if tf.get_default_session() is None:
        _ED_SESSION = tf.InteractiveSession()
    else:
        _ED_SESSION = tf.get_default_session()

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

    Raises
    ------
    InvalidArgumentError
        If the inputs have Inf or NaN values.
    """
    dependencies = [tf.verify_tensor_all_finite(y, msg='')]
    dependencies.extend([tf.verify_tensor_all_finite(x, msg='') for x in xs])

    with tf.control_dependencies(dependencies):
        # Calculate flattened vector grad_{xs} y.
        grads = tf.gradients(y, xs)
        grads = [tf.reshape(grad, [-1]) for grad in grads]
        grads = tf.concat(0, grads)
        # Loop over each element in the vector.
        mat = []
        d = grads.get_shape()[0]
        if not isinstance(d, int):
            d = grads.eval().shape[0]

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


def kl_multivariate_normal(loc_one, scale_one, loc_two=0.0, scale_two=1.0):
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

    Raises
    ------
    InvalidArgumentError
        If the location variables have Inf or NaN values, or if the scale
        variables are not positive.
    """
    dependencies = [tf.verify_tensor_all_finite(loc_one, msg=''),
                  tf.verify_tensor_all_finite(loc_two, msg=''),
                  tf.assert_positive(scale_one),
                  tf.assert_positive(scale_two)]
    loc_one = control_flow_ops.with_dependencies(dependencies, loc_one)
    loc_two = control_flow_ops.with_dependencies(dependencies, loc_two)
    scale_one = control_flow_ops.with_dependencies(dependencies, scale_one)
    scale_two = control_flow_ops.with_dependencies(dependencies, scale_two)

    if loc_two == 0.0 and scale_two == 1.0:
        return 0.5 * tf.reduce_sum(
            tf.square(scale_one) + tf.square(loc_one) - \
            1.0 - 2.0 * tf.log(scale_one))
    else:
        return 0.5 * tf.reduce_sum(
            tf.square(scale_one/scale_two) + \
            tf.square((loc_two - loc_one)/scale_two) - \
            1.0 + 2.0 * tf.log(scale_two) - 2.0 * tf.log(scale_one))


def log_mean_exp(x):
    """Compute the ``log_mean_exp`` of the elements in x.

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
    
    Raises
    ------
    InvalidArgumentError
        If the input has Inf or NaN values.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg='')]
    x = control_flow_ops.with_dependencies(dependencies, x)

    x_max = tf.reduce_max(x)
    return tf.add(x_max, tf.log(tf.reduce_mean(tf.exp(tf.sub(x, x_max)))))  


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
    
    Raises
    ------
    InvalidArgumentError
        If the input has Inf or NaN values.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg='')]
    x = control_flow_ops.with_dependencies(dependencies, x);

    x_max = tf.reduce_max(x)
    return tf.add(x_max, tf.log(tf.reduce_sum(tf.exp(tf.sub(x, x_max)))))


def logit(x):
    """Evaluate :math:`\log(x / (1 - x))` elementwise.

    Parameters
    ----------
    x : tf.Tensor
        scalar, vector, matrix, or n-Tensor

    Returns
    -------
    tf.Tensor
        size corresponding to size of input

    Raises
    ------
    InvalidArgumentError
        If the input is not between :math:`(0,1)` elementwise.
    """
    dependencies = [tf.assert_positive(x),
                    tf.assert_less(x, 1.0)]
    x = control_flow_ops.with_dependencies(dependencies, x)

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

    Raises
    ------
    InvalidArgumentError
        If the mean variables have Inf or NaN values, or if the scale
        and length variables are not positive.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                    tf.verify_tensor_all_finite(y, msg=''),
                    tf.assert_positive(sigma),
                    tf.assert_positive(l)]
    x = control_flow_ops.with_dependencies(dependencies, x)
    y = control_flow_ops.with_dependencies(dependencies, y)
    sigma = control_flow_ops.with_dependencies(dependencies, sigma)
    l = control_flow_ops.with_dependencies(dependencies, l)

    return tf.pow(sigma, 2.0) * \
            tf.exp(-1.0/(2.0*tf.pow(l, 2.0)) * \
            tf.reduce_sum(tf.pow(x - y , 2.0)))


def rbf(x, y=0.0, sigma=1.0, l=1.0):
    """Squared-exponential kernel element-wise

    .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) (x - y)^2 }

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
    
    Raises
    ------
    InvalidArgumentError
        If the mean variables have Inf or NaN values, or if the scale
        and length variables are not positive.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg=''),
                  tf.assert_positive(sigma),
                  tf.assert_positive(l)]
    x = control_flow_ops.with_dependencies(dependencies, x)
    y = control_flow_ops.with_dependencies(dependencies, y)
    sigma = control_flow_ops.with_dependencies(dependencies, sigma)
    l = control_flow_ops.with_dependencies(dependencies, l)

    return tf.pow(sigma, 2.0) * \
            tf.exp(-1.0/(2.0*tf.pow(l, 2.0)) * tf.pow(x - y , 2.0))


def set_seed(x):
    """Set seed for both NumPy and TensorFlow.

    Parameters
    ----------
    x : int, float
        seed
    """
    np.random.seed(x)
    tf.set_random_seed(x)


def softplus(x):
    """Elementwise Softplus function

    .. math:: \log(1 + \exp(x))

    If input `x < -30`, returns `0.0` exactly.

    If input `x > 30`, returns `x` exactly.

    TensorFlow can't currently autodiff through ``tf.nn.softplus()``.

    Parameters
    ----------
    x : tf.Tensor
        scalar, vector, matrix, or n-Tensor

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    
    Raises
    ------
    InvalidArgumentError
        If the input has Inf or NaN values.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg='')]
    x = control_flow_ops.with_dependencies(dependencies, x)

    result = tf.log(1.0 + tf.exp(x))

    less_than_thirty = tf.less(x, -30.0)
    result = tf.select(less_than_thirty, tf.zeros_like(x), result)

    greater_than_thirty = tf.greater(x, 30.0)
    result = tf.select(greater_than_thirty, x, result)

    return result


def stop_gradient(x):
    """Apply ``tf.stop_gradient()`` element-wise.

    Parameters
    ----------
    x : tf.Tensor or list
        scalar, vector, matrix, or n-Tensor or list thereof

    Returns
    -------
    tf.Tensor or list
        size corresponding to size of input
    """
    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        return tf.stop_gradient(x)
    else: # list
        return [tf.stop_gradient(i) for i in x]


def to_simplex(x):
    """Transform real vector of length ``(K-1)`` to a simplex of dimension ``K``
    using a backward stick breaking construction.

    Parameters
    ----------
    x : tf.tensor or np.array
        vector, or matrix

    Returns
    -------
    tf.Tensor
        Same shape as input but with last dimension of size ``K``.

    Raises
    ------
    InvalidArgumentError
        If the input has Inf or NaN values.

    Notes
    -----
    x as a 3d or higher tensor is not guaranteed to be supported.
    """
    dependencies = [tf.verify_tensor_all_finite(x, msg='')]
    x = control_flow_ops.with_dependencies(dependencies, x)

    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        shape = get_dims(x)
    else:
        shape = x.shape

    if len(shape) == 1:
        n_rows = ()
        K_minus_one = shape[0]
        eq = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one),
                             dtype=tf.float32))
        z = tf.sigmoid(eq + x)
        pil = tf.concat(0, [z, tf.constant([1.0])])
        piu = tf.concat(0, [tf.constant([1.0]), 1.0 - z])
        S = cumprod(piu)
        return S * pil
    else:
        n_rows = shape[0]
        K_minus_one = shape[1]
        eq = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one),
                             dtype=tf.float32))
        z = tf.sigmoid(eq + x)
        pil = tf.concat(1, [z, tf.ones([n_rows, 1])])
        piu = tf.concat(1, [tf.ones([n_rows, 1]), 1.0 - z])
        # cumulative product along 1st axis
        S = tf.pack([cumprod(piu_x) for piu_x in tf.unpack(piu)])
        return S * pil
