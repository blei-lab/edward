#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Normal
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf

from edward.models import Model, Normal
from edward.util import get_dims

def build_reparam_loss(self):
    """
    Loss function to minimize, whose gradient is a stochastic
    gradient based on the reparameterization trick.
    (Kingma and Welling, 2014)

    ELBO = E_{q(z; lambda)} [ log p(x, z) - log q(z; lambda) ]
    """
    self.zs = self.variational.sample(self.n_minibatch)
    z = self.zs
    # In order to call the probability model's log_prob(), swap key in
    # variational's sampled dictionary according to its correspondence
    # to posterior latent variables.
    xz = self.data # concatenate along with the data dictionary
    for key, value in self.zs.iteritems():
        xz[self.mapping[key]] = value

    p_log_prob = self.model.log_prob(xz)
    q_log_prob = self.variational.log_prob(z)
    self.loss = tf.reduce_mean(p_log_prob - q_log_prob)
    # TODO given the mapping's, we technically don't need any
    # container for either probability model or variational model
    return -self.loss

def log_prob(self, data_dict):
    # TODO can data_dict be thought of as a feed_dict?
    """
    Parameters
    ----------
    data_dict : dict
        Dictionary which binds all random variables (distribution
        objects) in the model (container object) to realizations
        (tf.Tensor or np.ndarray's). For each random variable of
        dimensions `shape`, its corresponding realization has either
        dimensions `shape` or `batch x shape`. Any optional outer
        dimension `batch` must be the same number for the optional
        outer dimension of each realization.

    Returns
    -------
    tf.Tensor
        If there is an outer dimension batch for at least one
        realization, return object is a vector of batch elements,
        evaluating the log density for each relization in that set of
        realizations and vectorize-summing over the reset. Otherwise a
        scalar.

    Notes
    -----
    This method may be removed in the future in favor of indexable
    log_prob methods, e.g., for automatic Rao-Blackwellization.

    This method assumes length of data_dict == length of self.layers and
    each item corresponds to a layer in self.layers.
    """
    # Get batch size from the first item in the dictionary.
    if isinstance(data_dict.values()[0], tf.Tensor):
        shape = get_dims(data_dict.values()[0])
    else: # NumPy array
        shape = data_dict.values()[0].shape

    n_minibatch = shape[0]
    log_prob = tf.zeros([n_minibatch], dtype=tf.float32)
    for layer in self.layers:
        # TODO does this work?
        log_prob += layer.log_prob(data_dict[layer])

    return log_prob

def sample(self, size=1):
    """
    Draws a mix of tensors and placeholders, corresponding to
    TensorFlow-based samplers and SciPy-based samplers depending
    on the layer.

    Parameters
    ----------
    size : int, optional
        Number of samples to draw.

    Returns
    -------
    dict
        Dictionary of distribution objects in the container assigned
        to a tf.Tensor. Each tf.Tensor is of size size x shape.
    """
    samples = {}
    for layer in self.layers:
        if layer.sample_tensor:
            samples[layer] = layer.sample(size)
        else:
            samples[layer] = tf.placeholder(tf.float32, (size, ) + layer.shape)

    # TODO what to return if only one layer, a dictionary of one
    # element?
    return samples

def np_dict(self, samples):
    """
    Form dictionary to feed any placeholders with np.array
    samples.

    Parameters
    ----------
    samples : dict
        Dictionary of distribution objects in the container assigned
        to a tf.Tensor. Each tf.Tensor is of size batch x shape.

    Return
    ------
    dict
        Dictionary of tf.placeholders in `samples` binded to SciPy
        samples.

    Notes
    -----
    This method assumes each samples[l] in samples has the same
    batch size, i.e., dimensions (batch x shape) for fixed batch
    and varying shape.
    """
    size = get_dims(samples.values()[0])[0]
    feed_dict = {}
    for layer, sample in samples.iteritems():
        if sample.name.startswith('Placeholder'):
            feed_dict[sample] = layer.sample(size)

    # TODO technically this doesn't require anything from self
    return feed_dict

ed.MFVI.build_reparam_loss = build_reparam_loss
Model.log_prob = log_prob
Model.sample = sample
Model.np_dict = np_dict

ed.set_seed(42)

mu = tf.constant([1.0])
std = tf.constant([1.0])
z = Normal(1, loc=mu, scale=std)

model = Model()
model.add(z)

qz = Normal()

variational = Model()
variational.add(qz)

# hard-code the mapping
ed.MFVI.mapping = {qz: z}
inference = ed.MFVI(model, variational, data={})
inference.run(n_iter=10000)
