#!/usr/bin/env python
"""
Generate `toy_data.tfrecords`.
"""
import numpy as np
import tensorflow as tf

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
writer = tf.python_io.TFRecordWriter("toy_data.tfrecords")
for x in xs:
    # Construct an Example proto object.
    example = tf.train.Example(
        # Example contains a Features proto object.
        features=tf.train.Features(
          # Features contains a dict of string to Feature proto objects
          feature={
            'outcome': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[x]))
    }))
    # Serialize the example proto object to a string.
    serialized = example.SerializeToString()
    # Write the serialized object to disk.
    writer.write(serialized)
