import tensorflow as tf

def predictive_check(T, data, model, latent, size=100, sess=tf.Session()):
    """
    Predictive check.
    It is a prior predictive check if latent is the prior and a
    posterior predictive check if latent is the posterior.
    (Box, 1980; Gelman, Meng, and Stern, 1996)

    It form an empirical distribution for the predictive discrepancy,
    q(T) = \int p(T(x) | z) q(z) dz
    by drawing replicated data sets xrep and calculating T(xrep) for
    each data set. Then it compares it to T(xobs).

    Parameters
    ----------
    T : function
        Test statistic.
    data : Data
        Observed data to check to.
    model : class with 'sample_likelihood' method
        model with likelihood distribution p(x | z) to sample from
    latent : class with 'sample' method
        latent variable distribution q(z) to sample from
    size : int, optional
        number of replicated data sets

    Returns
    -------
    np.ndarray
        Vector of size elements, (T(xrep^1), ..., T(xrep^size))
    """
    xobs = sess.run(data.data) # TODO generalize to arbitrary data
    Txobs = T(xobs)
    N = len(xobs) # TODO len, or shape[0]

    # TODO
    # size in variational sample
    # whether the sample method requires sess
    zreps = latent.sample([size, 1], sess)
    xreps = [model.sample_likelihood(zrep, N) for zrep in zreps]
    Txreps = [T(xrep) for xrep in xreps]
    return Txobs, Txreps
