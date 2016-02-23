class Data:
    """
    Base class for data.

    By default, it assumes the data is i.i.d. (independent and
    identically distributed). Use one of the derived classes for
    subsampling more complex data structures.

    Arguments
    ----------
    data: dict, tf.tensor, np.ndarray (TODO)
    n_minibatch: int, optional
        Number of samples for data subsampling. Default is to use all
        the data.
    """
    def __init__(self, data=None, n_minibatch=None):
        # TODO
        # self.data should be the right data structure
        # for calculating log prob's in any of the model wrappers but
        # be generic enough to enable our subsampling routines to work
        self.data = data
        self.n_minibatch = n_minibatch

    def sample(self):
        if self.n_minibatch is None:
            return self.data

        # TODO
        # subsample self.data uniformly with size self.n_minibatch
        return self.data
