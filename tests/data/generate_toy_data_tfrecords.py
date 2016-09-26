#!/usr/bin/env python
"""
Generate `toy_data.tfrecords`.
"""
import numpy as np
import tensorflow as tf

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
writer = tf.python_io.TFRecordWriter("toy_data.tfrecords")
for x in xs:
  example = tf.train.Example(features=tf.train.Features(
      feature={'outcome':
               tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))}))
  serialized = example.SerializeToString()
  writer.write(serialized)

writer.close()
