# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A quick hack to try deconv out."""

import collections

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

from prettytensor import layers
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import PAD_SAME
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PROVIDED

# pylint: disable=redefined-outer-name,invalid-name
@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'stddev', 'batch_normalize'))
class deconv2d(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               depth,
               name=PROVIDED,
               stride=None,
               activation_fn=None,
               l2loss=None,
               init=None,
               stddev=None,
               bias=True,
               edges=PAD_SAME,
               batch_normalize=False):
    """Adds a convolution to the stack of operations.

    The current head must be a rank 4 Tensor.

    Args:
      input_layer: The chainable object, supplied.
      kernel: The size of the patch for the pool, either an int or a length 1 or
        2 sequence (if length 1 or int, it is expanded).
      depth: The depth of the new Tensor.
      name: The name for this operation is also used to create/find the
        parameter variables.
      stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
        int, length 1 or 2, the stride in the first and last dimensions are 1.
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      init: An optional initialization. If not specified, uses Xavier
        initialization.
      stddev: A standard deviation to use in parameter initialization.
      bias: Set to False to not have a bias.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Set to True to batch_normalize this layer.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If head is not a rank 4 tensor or the  depth of the input
        (4th dim) is not known.
    """
    if len(input_layer.shape) != 4:
      raise ValueError(
          'Cannot perform conv2d on tensor with shape %s' % input_layer.shape)
    if input_layer.shape[3] is None:
      raise ValueError('Input depth must be known')
    kernel = _kernel(kernel)
    stride = _stride(stride)
    size = [kernel[0], kernel[1], depth, input_layer.shape[3]]

    books = input_layer.bookkeeper
    if init is None:
      if stddev is None:
        patch_size = size[0] * size[1]
        init = layers.xavier_init(size[2] * patch_size, size[3] * patch_size)
      elif stddev:
        init = tf.truncated_normal_initializer(stddev=stddev)
      else:
        init = tf.zeros_initializer
    elif stddev is not None:
      raise ValueError('Do not set both init and stddev.')
    dtype = input_layer.tensor.dtype
    params = self.variable('weights', size, init, dt=dtype)
    
    input_height = input_layer.shape[1]
    input_width = input_layer.shape[2]
    
    filter_height = kernel[0]
    filter_width = kernel[1]

    row_stride = stride[1]
    col_stride = stride[2]
    
    out_rows, out_cols = get2d_deconv_output_size(input_height, input_width, filter_height,
                               filter_width, row_stride, col_stride, edges)

    output_shape = [input_layer.shape[0], out_rows, out_cols, depth]
    y = tf.nn.conv2d_transpose(input_layer, params, output_shape, stride, edges)
    layers.add_l2loss(books, params, l2loss)
    if bias:
      y += self.variable(
          'bias',
          [size[-2]],
          tf.zeros_initializer,
          dt=dtype)
    books.add_scalar_summary(
        tf.reduce_mean(
            layers.spatial_slice_zeros(y)), '%s/zeros_spatial' % y.op.name)
    if batch_normalize:
      y = input_layer.with_tensor(y).batch_normalize()
    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(
          books,
          y,
          activation_fn[0],
          activation_args=activation_fn[1:])
    return input_layer.with_tensor(y)
# pylint: enable=redefined-outer-name,invalid-name

# Helper methods

def get2d_deconv_output_size(input_height, input_width, filter_height,
                           filter_width, row_stride, col_stride, padding_type):
    """Returns the number of rows and columns in a convolution/pooling output."""
    input_height = tensor_shape.as_dimension(input_height)
    input_width = tensor_shape.as_dimension(input_width)
    filter_height = tensor_shape.as_dimension(filter_height)
    filter_width = tensor_shape.as_dimension(filter_width)
    row_stride = int(row_stride)
    col_stride = int(col_stride)

    # Compute number of rows in the output, based on the padding.
    if input_height.value is None or filter_height.value is None:
      out_rows = None
    elif padding_type == "VALID":
      out_rows = (input_height.value - 1) * row_stride + filter_height.value 
    elif padding_type == "SAME":
      out_rows = input_height.value * row_stride
    else:
      raise ValueError("Invalid value for padding: %r" % padding_type)

    # Compute number of columns in the output, based on the padding.
    if input_width.value is None or filter_width.value is None:
      out_cols = None
    elif padding_type == "VALID":
      out_cols = (input_width.value - 1) * col_stride + filter_width.value
    elif padding_type == "SAME":
      out_cols = input_width.value * col_stride

    return out_rows, out_cols

def _kernel(kernel_spec):
  """Expands the kernel spec into a length 2 list.

  Args:
    kernel_spec: An integer or a length 1 or 2 sequence that is expanded to a
      list.
  Returns:
    A length 2 list.
  """
  if isinstance(kernel_spec, int):
    return [kernel_spec, kernel_spec]
  elif len(kernel_spec) == 1:
    return [kernel_spec[0], kernel_spec[0]]
  else:
    assert len(kernel_spec) == 2
    return kernel_spec


def _stride(stride_spec):
  """Expands the stride spec into a length 4 list.

  Args:
    stride_spec: None, an integer or a length 1, 2, or 4 sequence.
  Returns:
    A length 4 list.
  """
  if stride_spec is None:
    return [1, 1, 1, 1]
  elif isinstance(stride_spec, int):
    return [1, stride_spec, stride_spec, 1]
  elif len(stride_spec) == 1:
    return [1, stride_spec[0], stride_spec[0], 1]
  elif len(stride_spec) == 2:
    return [1, stride_spec[0], stride_spec[1], 1]
  else:
    assert len(stride_spec) == 4
    return stride_spec
