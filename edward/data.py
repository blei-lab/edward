import numpy as np
import tensorflow as tf

class Data:
    """
    Base class for data.

    By default, it assumes the data is an array and if requested will
    perform data subsampling according to slices of the first index
    (e.g., elements in a vector, rows in a matrix, y-by-z matrices in
    a x-by-y-by-z tensor). Use one of the derived classes for
    subsampling more complex data structures.

    Arguments
    ----------
    data: dict, tf.tensor, np.ndarray, optional
        Data whose type depends on the type of model it is fed into:
        Stan, TensorFlow, and NumPy/SciPy respectively.
    shuffled: bool, optional
        Whether the data is shuffled.

    Notes
    -----
    Data subsampling is not currently available for Stan models.
    """
    def __init__(self, data=None, shuffled=True):
        self.data = data
        if not shuffled:
            # TODO
            # shuffle self.data
            raise NotImplementedError()

        self.counter = 0
        if self.data is None:
            self.N = None
        elif isinstance(self.data, tf.Tensor):
            self.N = self.data.get_shape()[0].value
        elif isinstance(self.data, np.ndarray):
            self.N = self.data.shape[0]
        elif isinstance(self.data, dict):
            pass
        else:
            raise

    def sample(self, n_data=None):
        # TODO scale gradient and printed loss by self.N / self.n_data
        if n_data is None:
            return self.data

        counter_new = self.counter + n_data
        if isinstance(self.data, tf.Tensor):
            if counter_new <= self.N:
                minibatch = tf.gather(self.data,
                                      list(range(self.counter, counter_new)))
                self.counter = counter_new
            else:
                counter_new = counter_new - self.N
                minibatch = tf.gather(self.data,
                                      list(range(self.counter, self.N)) + \
                                      list(range(0, counter_new)))
                self.counter = counter_new

            return minibatch
        elif isinstance(self.data, np.ndarray):
            if counter_new <= self.N:
                minibatch = self.data[self.counter:counter_new]
                self.counter = counter_new
            else:
                counter_new = counter_new - self.N
                minibatch = np.concatenate((self.data[self.counter:],
                                            self.data[:counter_new]))
                self.counter = counter_new

            return minibatch
        else:
            minibatch = self.data.copy()
            if counter_new <= self.N:
                minibatch['y'] = minibatch['y'][self.counter:counter_new]
                self.counter = counter_new
            else:
                counter_new = counter_new - self.N
                minibatch['y'] = minibatch['y'][self.counter:] + \
                                 minibatch['y'][:counter_new]
                self.counter = counter_new

            return minibatch
