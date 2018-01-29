"""Generate `toy_data.tfrecords`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def main(_):
  xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
  writer = tf.python_io.TFRecordWriter("toy_data.tfrecords")
  for x in xs:
    example = tf.train.Example(features=tf.train.Features(
        feature={'outcome':
                 tf.train.Feature(float_list=tf.train.FloatList(value=[x]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)

  writer.close()

if __name__ == "__main__":
  tf.app.run()
