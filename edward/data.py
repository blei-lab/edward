from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class DataGenerator(object):
    """Base class for Edward data generators.

    Its only method is next(), which returns a batch of data. By
    default, the method returns all the data. If requested, a batch is
    given by data subsampling according to slices of the first index
    (e.g., elements in a vector, rows in a matrix, y-by-z matrices in
    a x-by-y-by-z tensor). Use one of the derived classes for
    subsampling more complex data structures.

    Internally, ``self.counter`` stores the last accessed data index.
    It is used to obtain the next batch of data starting from
    ``self.counter`` to the size of the data set.
    """
    def __init__(self, data):
        """Initialization.

        Parameters
        ----------
        data : np.ndarray
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            raise NotImplementedError()

        self.N = data.shape[0]
        self.counter = 0

    def next(self, n_data=None):
        """Data sampling method.

        At any given point, the internal counter ``self.counter`` tracks the
        last datapoint returned by ``next``.

        If the requested number of datapoints ``n_data`` goes beyond the size
        of the dataset, the internal counter wraps around the size of the
        dataset. The returned batch, thus, may include datapoints from the
        beginning of the dataset.

        Parameters
        ----------
        n_data : int, optional
            Number of datapoints to sample.

            Defaults to total number of datapoints in object.

        Returns
        -------
        np.ndarray
            Array whose values are all subsampled.
        """
        if n_data is None:
            return self.data

        counter_old = self.counter
        counter_new = counter_old + n_data
        if counter_new <= self.N:
            batch = self.data[counter_old:counter_new]
        else:
            counter_new = counter_new - self.N
            batch = np.concatenate((self.data[counter_old:],
                                    self.data[:counter_new]))

        self.counter = counter_new
        return batch
