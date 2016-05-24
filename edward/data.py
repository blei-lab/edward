import numpy as np
import tensorflow as tf

class Data:
    """
    Base class for data.

    By default, it assumes the data is an array (or list of arrays).
    If requested will perform data subsampling according to slices of
    the first index (e.g., elements in a vector, rows in a matrix,
    y-by-z matrices in a x-by-y-by-z tensor). Use one of the derived
    classes for subsampling more complex data structures.

    Arguments
    ----------
    data: tf.tensor, np.ndarray, list, dict, optional
        Data whose type depends on the type of model it is fed into.
        If TensorFlow, must be tf.tensor or list (see notes).
        If Stan, must be dict.
        If PyMC3, must be np.ndarray.
        If NumPy/SciPy, must be np.ndarray or list of np.ndarrays.
    shuffled: bool, optional
        Whether the data is shuffled.

    Notes
    -----
    For TensorFlow models, data argument can be list of placeholders
    or list of np.ndarrays. If np.ndarrays, it will use mini-batches
    of the np.arrays during computation. If placeholders, user must
    manually control mini-batches and feed in the placeholders.

    Data subsampling is not currently available for Stan models.

    Internally, self.counter stores the last accessed data index. It
    is used to obtain the next batch of data starting from
    self.counter to the size of the data set.
    """
    def __init__(self, data=None, shuffled=True):
        self.data = data
        if not shuffled:
            # TODO
            # shuffle self.data
            raise NotImplementedError()

        if self.data is None:
            pass
        elif isinstance(self.data, tf.Tensor):
            self.N = self.data.get_shape()[0].value
            self.counter = 0
        elif isinstance(self.data, np.ndarray):
            self.N = self.data.shape[0]
            self.counter = 0
        elif isinstance(self.data, list):
            if isinstance(self.data[0], np.ndarray):
                self.N = [x.shape[0] for x in self.data]
                self.counter = [0]*len(self.data)
            else: # list of placeholders
                # need data set size to scale gradients appropriately
                pass
        elif isinstance(self.data, dict):
            pass
        else:
            raise NotImplementedError()

    def sample(self, n_data=None):
        # TODO scale gradient and printed loss by self.N / self.n_data
        if n_data is None or self.data is None:
            return self.data

        if isinstance(self.data, tf.Tensor):
            counter_new = self.counter + n_data
            if counter_new <= self.N:
                minibatch = tf.gather(self.data,
                                      list(range(self.counter, counter_new)))
            else:
                counter_new = counter_new - self.N
                minibatch = tf.gather(self.data,
                                      list(range(self.counter, self.N)) + \
                                      list(range(0, counter_new)))

            self.counter = counter_new
            return minibatch
        elif isinstance(self.data, np.ndarray):
            counter_new = self.counter + n_data
            if counter_new <= self.N:
                minibatch = self.data[self.counter:counter_new]
            else:
                counter_new = counter_new - self.N
                minibatch = np.concatenate((self.data[self.counter:],
                                            self.data[:counter_new]))

            self.counter = counter_new
            return minibatch
        elif isinstance(self.data, list):
            if isinstance(self.data[0], np.ndarray):
                minibatch = [0]*len(self.data)
                for i in range(len(self.data)):
                    counter_new = self.counter[i] + n_data
                    if counter_new <= self.N[i]:
                        minibatch[i] = self.data[i][self.counter[i]:counter_new]
                    else:
                        counter_new = counter_new - self.N[i]
                        minibatch[i] = np.concatenate((self.data[i][self.counter[i]:],
                                                       self.data[i][:counter_new]))

                    self.counter[i] = counter_new

                return minibatch
            else: # list of placeholders
                raise NotImplementedError()
        else: # dict
            raise NotImplementedError()
