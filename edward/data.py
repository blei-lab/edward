from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import get_dims


class DataGenerator(object):
    """Base class for Edward data generators.

    Its only method is next(), which returns a batch of data. By
    default, the method returns all the data. If requested, batches
    are given by data subsampling all values in the dictionary
    according to slices of the first index (e.g., elements in a
    vector, rows in a matrix, y-by-z matrices in a x-by-y-by-z
    tensor). Use one of the derived classes for subsampling more
    complex data structures.

    Data subsampling is not currently available for Stan models.

    Internally, ``self.counter`` stores the last accessed data index. It
    is used to obtain the next batch of data starting from
    ``self.counter`` to the size of the data set.
    """
    def __init__(self, data=None):
        """Initialization.

        Parameters
        ----------
        data : dict of np.ndarray's, optional
            Dictionary which binds named keys of data to their values.
            TODO or tf.placeholder for TensorFlow
        """
        if data is None:
            self.data = {}
        elif isinstance(data, dict):
            self.data = data
        else:
            raise NotImplementedError()

        self.N = {}
        self.counter = {}
        for key, value in self.data.items():
            self.N[key] = get_dims(value)[0]
            self.counter[key] = 0

    def next(self, xs):
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
            Number of datapoints to sample

            Defaults to total number of datapoints in object.

        Returns
        -------
        dict of np.ndarray's
            Dictionary whose values are all subsampled.

        Notes
        -----
        If data dictionary has any tf.placeholder's, the user must
        control batches by feeding in the tf.placeholder's manually.
        In such a case, always use this method with ``n_data`` set to
        None.
        """
        #if n_data is None or not self.data: # n_data=None or empty dictionary
        #    return self.data

        batch = {}
        for key, value in self.data.items():
            N = self.N[key]
            n_data = get_dims(xs[key])[0] # TODO
            if n_data is None:
                n_data = N

            counter_old = self.counter[key]
            if isinstance(value, np.ndarray):
                counter_new = counter_old + n_data
                if counter_new <= N:
                    batch_value = value[counter_old:counter_new]
                else:
                    counter_new = counter_new - N
                    batch_value = np.concatenate((value[counter_old:],
                                                value[:counter_new]))
            else:
                raise NotImplementedError()

            self.counter[key] = counter_new
            batch[xs[key]] = batch_value

        return batch

    def make_placeholders(self, n_data=None):
        placeholder_dict = {}
        for key, value in self.data.items():
            # TODO maybe not best place
            if isinstance(value, tf.Tensor):
                if value.name.startswith('Placeholder'):
                    placeholder_dict[key] = value
            else:
                placeholder_dict[key] = tf.placeholder(tf.float32,
                                                       (n_data, ) + value.shape[1:])

        return placeholder_dict
        #return {key: tf.placeholder(tf.float32, (n_data, ) + value.shape[1:])
        #        for key, value in self.data.items()}
