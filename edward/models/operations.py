from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.models.random_variables import DelayedOperation

import six
import tensorflow as tf


class add_to_collection(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_to_collection, self).__init__(tf.add_to_collection, *args, **kwargs)


class as_dtype(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(as_dtype, self).__init__(tf.as_dtype, *args, **kwargs)


class bytes(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(bytes, self).__init__(tf.bytes, *args, **kwargs)


class container(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(container, self).__init__(tf.container, *args, **kwargs)


class control_dependencies(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(control_dependencies, self).__init__(tf.control_dependencies, *args, **kwargs)


class convert_to_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convert_to_tensor, self).__init__(tf.convert_to_tensor, *args, **kwargs)


class convert_to_tensor_or_indexed_slices(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convert_to_tensor_or_indexed_slices, self).__init__(tf.convert_to_tensor_or_indexed_slices, *args, **kwargs)


class device(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(device, self).__init__(tf.device, *args, **kwargs)


class DeviceSpec(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DeviceSpec, self).__init__(tf.DeviceSpec, *args, **kwargs)


class Dimension(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Dimension, self).__init__(tf.Dimension, *args, **kwargs)


class DType(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DType, self).__init__(tf.DType, *args, **kwargs)


class get_collection(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_collection, self).__init__(tf.get_collection, *args, **kwargs)


class get_collection_ref(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_collection_ref, self).__init__(tf.get_collection_ref, *args, **kwargs)


class get_default_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_default_graph, self).__init__(tf.get_default_graph, *args, **kwargs)


class get_seed(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_seed, self).__init__(tf.get_seed, *args, **kwargs)


class Graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(tf.Graph, *args, **kwargs)


class GraphKeys(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(GraphKeys, self).__init__(tf.GraphKeys, *args, **kwargs)


class import_graph_def(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(import_graph_def, self).__init__(tf.import_graph_def, *args, **kwargs)


class load_file_system_library(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(load_file_system_library, self).__init__(tf.load_file_system_library, *args, **kwargs)


class load_op_library(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(load_op_library, self).__init__(tf.load_op_library, *args, **kwargs)


class name_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(name_scope, self).__init__(tf.name_scope, *args, **kwargs)


class NoGradient(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(NoGradient, self).__init__(tf.NoGradient, *args, **kwargs)


class op_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(op_scope, self).__init__(tf.op_scope, *args, **kwargs)


class Operation(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Operation, self).__init__(tf.Operation, *args, **kwargs)


class register_tensor_conversion_function(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(register_tensor_conversion_function, self).__init__(tf.register_tensor_conversion_function, *args, **kwargs)


class RegisterGradient(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RegisterGradient, self).__init__(tf.RegisterGradient, *args, **kwargs)


class RegisterShape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RegisterShape, self).__init__(tf.RegisterShape, *args, **kwargs)


class reset_default_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reset_default_graph, self).__init__(tf.reset_default_graph, *args, **kwargs)


class Tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Tensor, self).__init__(tf.Tensor, *args, **kwargs)


class TensorShape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorShape, self).__init__(tf.TensorShape, *args, **kwargs)


class assert_equal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_equal, self).__init__(tf.assert_equal, *args, **kwargs)


class assert_integer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_integer, self).__init__(tf.assert_integer, *args, **kwargs)


class assert_less(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_less, self).__init__(tf.assert_less, *args, **kwargs)


class assert_less_equal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_less_equal, self).__init__(tf.assert_less_equal, *args, **kwargs)


class assert_negative(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_negative, self).__init__(tf.assert_negative, *args, **kwargs)


class assert_non_negative(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_non_negative, self).__init__(tf.assert_non_negative, *args, **kwargs)


class assert_non_positive(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_non_positive, self).__init__(tf.assert_non_positive, *args, **kwargs)


class assert_positive(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_positive, self).__init__(tf.assert_positive, *args, **kwargs)


class assert_proper_iterable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_proper_iterable, self).__init__(tf.assert_proper_iterable, *args, **kwargs)


class assert_rank(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_rank, self).__init__(tf.assert_rank, *args, **kwargs)


class assert_rank_at_least(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_rank_at_least, self).__init__(tf.assert_rank_at_least, *args, **kwargs)


class assert_type(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_type, self).__init__(tf.assert_type, *args, **kwargs)


class is_non_decreasing(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_non_decreasing, self).__init__(tf.is_non_decreasing, *args, **kwargs)


class is_numeric_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_numeric_tensor, self).__init__(tf.is_numeric_tensor, *args, **kwargs)


class is_strictly_increasing(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_strictly_increasing, self).__init__(tf.is_strictly_increasing, *args, **kwargs)


class constant(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(constant, self).__init__(tf.constant, *args, **kwargs)


class fill(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(fill, self).__init__(tf.fill, *args, **kwargs)


class linspace(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(linspace, self).__init__(tf.linspace, *args, **kwargs)


class multinomial(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(multinomial, self).__init__(tf.multinomial, *args, **kwargs)


class ones(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ones, self).__init__(tf.ones, *args, **kwargs)


class ones_like(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ones_like, self).__init__(tf.ones_like, *args, **kwargs)


class random_crop(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_crop, self).__init__(tf.random_crop, *args, **kwargs)


class random_gamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_gamma, self).__init__(tf.random_gamma, *args, **kwargs)


class random_normal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_normal, self).__init__(tf.random_normal, *args, **kwargs)


class random_shuffle(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_shuffle, self).__init__(tf.random_shuffle, *args, **kwargs)


class random_uniform(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_uniform, self).__init__(tf.random_uniform, *args, **kwargs)


class range(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(range, self).__init__(tf.range, *args, **kwargs)


class set_random_seed(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(set_random_seed, self).__init__(tf.set_random_seed, *args, **kwargs)


class truncated_normal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(truncated_normal, self).__init__(tf.truncated_normal, *args, **kwargs)


class zeros(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(zeros, self).__init__(tf.zeros, *args, **kwargs)


class zeros_like(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(zeros_like, self).__init__(tf.zeros_like, *args, **kwargs)


class all_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(all_variables, self).__init__(tf.all_variables, *args, **kwargs)


class assert_variables_initialized(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_variables_initialized, self).__init__(tf.assert_variables_initialized, *args, **kwargs)


class assign(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assign, self).__init__(tf.assign, *args, **kwargs)


class assign_add(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assign_add, self).__init__(tf.assign_add, *args, **kwargs)


class assign_sub(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assign_sub, self).__init__(tf.assign_sub, *args, **kwargs)


class constant_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(constant_initializer, self).__init__(tf.constant_initializer, *args, **kwargs)


class count_up_to(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(count_up_to, self).__init__(tf.count_up_to, *args, **kwargs)


class device(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(device, self).__init__(tf.device, *args, **kwargs)


class export_meta_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(export_meta_graph, self).__init__(tf.export_meta_graph, *args, **kwargs)


class get_checkpoint_state(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_checkpoint_state, self).__init__(tf.get_checkpoint_state, *args, **kwargs)


class get_variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_variable, self).__init__(tf.get_variable, *args, **kwargs)


class get_variable_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_variable_scope, self).__init__(tf.get_variable_scope, *args, **kwargs)


class import_meta_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(import_meta_graph, self).__init__(tf.import_meta_graph, *args, **kwargs)


class IndexedSlices(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(IndexedSlices, self).__init__(tf.IndexedSlices, *args, **kwargs)


class initialize_all_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(initialize_all_variables, self).__init__(tf.initialize_all_variables, *args, **kwargs)


class initialize_local_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(initialize_local_variables, self).__init__(tf.initialize_local_variables, *args, **kwargs)


class initialize_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(initialize_variables, self).__init__(tf.initialize_variables, *args, **kwargs)


class is_variable_initialized(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_variable_initialized, self).__init__(tf.is_variable_initialized, *args, **kwargs)


class latest_checkpoint(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(latest_checkpoint, self).__init__(tf.latest_checkpoint, *args, **kwargs)


class local_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(local_variables, self).__init__(tf.local_variables, *args, **kwargs)


class make_template(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(make_template, self).__init__(tf.make_template, *args, **kwargs)


class min_max_variable_partitioner(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(min_max_variable_partitioner, self).__init__(tf.min_max_variable_partitioner, *args, **kwargs)


class moving_average_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(moving_average_variables, self).__init__(tf.moving_average_variables, *args, **kwargs)


class no_regularizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(no_regularizer, self).__init__(tf.no_regularizer, *args, **kwargs)


class ones_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ones_initializer, self).__init__(tf.ones_initializer, *args, **kwargs)


class random_normal_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_normal_initializer, self).__init__(tf.random_normal_initializer, *args, **kwargs)


class random_uniform_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_uniform_initializer, self).__init__(tf.random_uniform_initializer, *args, **kwargs)


class report_uninitialized_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(report_uninitialized_variables, self).__init__(tf.report_uninitialized_variables, *args, **kwargs)


class Saver(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Saver, self).__init__(tf.Saver, *args, **kwargs)


class scatter_add(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(scatter_add, self).__init__(tf.scatter_add, *args, **kwargs)


class scatter_sub(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(scatter_sub, self).__init__(tf.scatter_sub, *args, **kwargs)


class scatter_update(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(scatter_update, self).__init__(tf.scatter_update, *args, **kwargs)


class sparse_mask(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_mask, self).__init__(tf.sparse_mask, *args, **kwargs)


class trainable_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(trainable_variables, self).__init__(tf.trainable_variables, *args, **kwargs)


class truncated_normal_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(truncated_normal_initializer, self).__init__(tf.truncated_normal_initializer, *args, **kwargs)


class uniform_unit_scaling_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(uniform_unit_scaling_initializer, self).__init__(tf.uniform_unit_scaling_initializer, *args, **kwargs)


class update_checkpoint_state(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(update_checkpoint_state, self).__init__(tf.update_checkpoint_state, *args, **kwargs)


class Variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Variable, self).__init__(tf.Variable, *args, **kwargs)


class variable_axis_size_partitioner(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(variable_axis_size_partitioner, self).__init__(tf.variable_axis_size_partitioner, *args, **kwargs)


class variable_op_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(variable_op_scope, self).__init__(tf.variable_op_scope, *args, **kwargs)


class variable_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(variable_scope, self).__init__(tf.variable_scope, *args, **kwargs)


class VariableScope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(VariableScope, self).__init__(tf.VariableScope, *args, **kwargs)


class zeros_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(zeros_initializer, self).__init__(tf.zeros_initializer, *args, **kwargs)


class batch_to_space(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_to_space, self).__init__(tf.batch_to_space, *args, **kwargs)


class bitcast(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(bitcast, self).__init__(tf.bitcast, *args, **kwargs)


class boolean_mask(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(boolean_mask, self).__init__(tf.boolean_mask, *args, **kwargs)


class cast(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cast, self).__init__(tf.cast, *args, **kwargs)


class concat(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(concat, self).__init__(tf.concat, *args, **kwargs)


class depth_to_space(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(depth_to_space, self).__init__(tf.depth_to_space, *args, **kwargs)


class dynamic_partition(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(dynamic_partition, self).__init__(tf.dynamic_partition, *args, **kwargs)


class dynamic_stitch(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(dynamic_stitch, self).__init__(tf.dynamic_stitch, *args, **kwargs)


class expand_dims(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(expand_dims, self).__init__(tf.expand_dims, *args, **kwargs)


class extract_image_patches(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_image_patches, self).__init__(tf.extract_image_patches, *args, **kwargs)


class gather(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(gather, self).__init__(tf.gather, *args, **kwargs)


class gather_nd(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(gather_nd, self).__init__(tf.gather_nd, *args, **kwargs)


class meshgrid(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(meshgrid, self).__init__(tf.meshgrid, *args, **kwargs)


class one_hot(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(one_hot, self).__init__(tf.one_hot, *args, **kwargs)


class pack(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(pack, self).__init__(tf.pack, *args, **kwargs)


class pad(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(pad, self).__init__(tf.pad, *args, **kwargs)


class rank(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(rank, self).__init__(tf.rank, *args, **kwargs)


class reshape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reshape, self).__init__(tf.reshape, *args, **kwargs)


class reverse(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reverse, self).__init__(tf.reverse, *args, **kwargs)


class reverse_sequence(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reverse_sequence, self).__init__(tf.reverse_sequence, *args, **kwargs)


class saturate_cast(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(saturate_cast, self).__init__(tf.saturate_cast, *args, **kwargs)


class shape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(shape, self).__init__(tf.shape, *args, **kwargs)


class shape_n(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(shape_n, self).__init__(tf.shape_n, *args, **kwargs)


class size(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(size, self).__init__(tf.size, *args, **kwargs)


class slice(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(slice, self).__init__(tf.slice, *args, **kwargs)


class space_to_batch(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(space_to_batch, self).__init__(tf.space_to_batch, *args, **kwargs)


class space_to_depth(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(space_to_depth, self).__init__(tf.space_to_depth, *args, **kwargs)


class split(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(split, self).__init__(tf.split, *args, **kwargs)


class squeeze(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(squeeze, self).__init__(tf.squeeze, *args, **kwargs)


class string_to_number(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(string_to_number, self).__init__(tf.string_to_number, *args, **kwargs)


class tile(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(tile, self).__init__(tf.tile, *args, **kwargs)


class to_bfloat16(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(to_bfloat16, self).__init__(tf.to_bfloat16, *args, **kwargs)


class to_double(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(to_double, self).__init__(tf.to_double, *args, **kwargs)


class to_float(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(to_float, self).__init__(tf.to_float, *args, **kwargs)


class to_int32(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(to_int32, self).__init__(tf.to_int32, *args, **kwargs)


class to_int64(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(to_int64, self).__init__(tf.to_int64, *args, **kwargs)


class transpose(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(transpose, self).__init__(tf.transpose, *args, **kwargs)


class unique_with_counts(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(unique_with_counts, self).__init__(tf.unique_with_counts, *args, **kwargs)


class unpack(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(unpack, self).__init__(tf.unpack, *args, **kwargs)


class abs(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(abs, self).__init__(tf.abs, *args, **kwargs)


class accumulate_n(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(accumulate_n, self).__init__(tf.accumulate_n, *args, **kwargs)


class acos(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(acos, self).__init__(tf.acos, *args, **kwargs)


class add(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add, self).__init__(tf.add, *args, **kwargs)


class add_n(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_n, self).__init__(tf.add_n, *args, **kwargs)


class argmax(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(argmax, self).__init__(tf.argmax, *args, **kwargs)


class argmin(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(argmin, self).__init__(tf.argmin, *args, **kwargs)


class asin(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(asin, self).__init__(tf.asin, *args, **kwargs)


class atan(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(atan, self).__init__(tf.atan, *args, **kwargs)


class batch_cholesky(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_cholesky, self).__init__(tf.batch_cholesky, *args, **kwargs)


class batch_cholesky_solve(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_cholesky_solve, self).__init__(tf.batch_cholesky_solve, *args, **kwargs)


class batch_fft(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_fft, self).__init__(tf.batch_fft, *args, **kwargs)


class batch_fft2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_fft2d, self).__init__(tf.batch_fft2d, *args, **kwargs)


class batch_fft3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_fft3d, self).__init__(tf.batch_fft3d, *args, **kwargs)


class batch_ifft(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_ifft, self).__init__(tf.batch_ifft, *args, **kwargs)


class batch_ifft2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_ifft2d, self).__init__(tf.batch_ifft2d, *args, **kwargs)


class batch_ifft3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_ifft3d, self).__init__(tf.batch_ifft3d, *args, **kwargs)


class batch_matmul(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matmul, self).__init__(tf.batch_matmul, *args, **kwargs)


class batch_matrix_band_part(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_band_part, self).__init__(tf.batch_matrix_band_part, *args, **kwargs)


class batch_matrix_determinant(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_determinant, self).__init__(tf.batch_matrix_determinant, *args, **kwargs)


class batch_matrix_diag(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_diag, self).__init__(tf.batch_matrix_diag, *args, **kwargs)


class batch_matrix_diag_part(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_diag_part, self).__init__(tf.batch_matrix_diag_part, *args, **kwargs)


class batch_matrix_inverse(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_inverse, self).__init__(tf.batch_matrix_inverse, *args, **kwargs)


class batch_matrix_set_diag(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_set_diag, self).__init__(tf.batch_matrix_set_diag, *args, **kwargs)


class batch_matrix_solve(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_solve, self).__init__(tf.batch_matrix_solve, *args, **kwargs)


class batch_matrix_solve_ls(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_solve_ls, self).__init__(tf.batch_matrix_solve_ls, *args, **kwargs)


class batch_matrix_transpose(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_transpose, self).__init__(tf.batch_matrix_transpose, *args, **kwargs)


class batch_matrix_triangular_solve(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_triangular_solve, self).__init__(tf.batch_matrix_triangular_solve, *args, **kwargs)


class batch_self_adjoint_eig(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_self_adjoint_eig, self).__init__(tf.batch_self_adjoint_eig, *args, **kwargs)


class ceil(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ceil, self).__init__(tf.ceil, *args, **kwargs)


class cholesky(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cholesky, self).__init__(tf.cholesky, *args, **kwargs)


class cholesky_solve(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cholesky_solve, self).__init__(tf.cholesky_solve, *args, **kwargs)


class complex(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(complex, self).__init__(tf.complex, *args, **kwargs)


class complex_abs(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(complex_abs, self).__init__(tf.complex_abs, *args, **kwargs)


class conj(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(conj, self).__init__(tf.conj, *args, **kwargs)


class cos(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cos, self).__init__(tf.cos, *args, **kwargs)


class cross(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cross, self).__init__(tf.cross, *args, **kwargs)


class cumprod(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cumprod, self).__init__(tf.cumprod, *args, **kwargs)


class cumsum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cumsum, self).__init__(tf.cumsum, *args, **kwargs)


class diag(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(diag, self).__init__(tf.diag, *args, **kwargs)


class diag_part(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(diag_part, self).__init__(tf.diag_part, *args, **kwargs)


class digamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(digamma, self).__init__(tf.digamma, *args, **kwargs)


class div(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(div, self).__init__(tf.div, *args, **kwargs)


class edit_distance(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(edit_distance, self).__init__(tf.edit_distance, *args, **kwargs)


class erf(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(erf, self).__init__(tf.erf, *args, **kwargs)


class erfc(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(erfc, self).__init__(tf.erfc, *args, **kwargs)


class exp(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(exp, self).__init__(tf.exp, *args, **kwargs)


class fft(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(fft, self).__init__(tf.fft, *args, **kwargs)


class fft2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(fft2d, self).__init__(tf.fft2d, *args, **kwargs)


class fft3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(fft3d, self).__init__(tf.fft3d, *args, **kwargs)


class floor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(floor, self).__init__(tf.floor, *args, **kwargs)


class floordiv(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(floordiv, self).__init__(tf.floordiv, *args, **kwargs)


class ifft(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ifft, self).__init__(tf.ifft, *args, **kwargs)


class ifft2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ifft2d, self).__init__(tf.ifft2d, *args, **kwargs)


class ifft3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ifft3d, self).__init__(tf.ifft3d, *args, **kwargs)


class igamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(igamma, self).__init__(tf.igamma, *args, **kwargs)


class igammac(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(igammac, self).__init__(tf.igammac, *args, **kwargs)


class imag(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(imag, self).__init__(tf.imag, *args, **kwargs)


class inv(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(inv, self).__init__(tf.inv, *args, **kwargs)


class invert_permutation(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(invert_permutation, self).__init__(tf.invert_permutation, *args, **kwargs)


class lbeta(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(lbeta, self).__init__(tf.lbeta, *args, **kwargs)


class lgamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(lgamma, self).__init__(tf.lgamma, *args, **kwargs)


class listdiff(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(listdiff, self).__init__(tf.listdiff, *args, **kwargs)


class log(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(log, self).__init__(tf.log, *args, **kwargs)


class matmul(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matmul, self).__init__(tf.matmul, *args, **kwargs)


class matrix_determinant(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matrix_determinant, self).__init__(tf.matrix_determinant, *args, **kwargs)


class matrix_inverse(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matrix_inverse, self).__init__(tf.matrix_inverse, *args, **kwargs)


class matrix_solve(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matrix_solve, self).__init__(tf.matrix_solve, *args, **kwargs)


class matrix_solve_ls(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matrix_solve_ls, self).__init__(tf.matrix_solve_ls, *args, **kwargs)


class matrix_triangular_solve(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matrix_triangular_solve, self).__init__(tf.matrix_triangular_solve, *args, **kwargs)


class maximum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(maximum, self).__init__(tf.maximum, *args, **kwargs)


class minimum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(minimum, self).__init__(tf.minimum, *args, **kwargs)


class mod(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(mod, self).__init__(tf.mod, *args, **kwargs)


class mul(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(mul, self).__init__(tf.mul, *args, **kwargs)


class neg(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(neg, self).__init__(tf.neg, *args, **kwargs)


class polygamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(polygamma, self).__init__(tf.polygamma, *args, **kwargs)


class pow(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(pow, self).__init__(tf.pow, *args, **kwargs)


class real(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(real, self).__init__(tf.real, *args, **kwargs)


class reduce_all(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_all, self).__init__(tf.reduce_all, *args, **kwargs)


class reduce_any(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_any, self).__init__(tf.reduce_any, *args, **kwargs)


class reduce_max(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_max, self).__init__(tf.reduce_max, *args, **kwargs)


class reduce_mean(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_mean, self).__init__(tf.reduce_mean, *args, **kwargs)


class reduce_min(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_min, self).__init__(tf.reduce_min, *args, **kwargs)


class reduce_prod(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_prod, self).__init__(tf.reduce_prod, *args, **kwargs)


class reduce_sum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_sum, self).__init__(tf.reduce_sum, *args, **kwargs)


class round(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(round, self).__init__(tf.round, *args, **kwargs)


class rsqrt(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(rsqrt, self).__init__(tf.rsqrt, *args, **kwargs)


class scalar_mul(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(scalar_mul, self).__init__(tf.scalar_mul, *args, **kwargs)


class segment_max(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(segment_max, self).__init__(tf.segment_max, *args, **kwargs)


class segment_mean(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(segment_mean, self).__init__(tf.segment_mean, *args, **kwargs)


class segment_min(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(segment_min, self).__init__(tf.segment_min, *args, **kwargs)


class segment_prod(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(segment_prod, self).__init__(tf.segment_prod, *args, **kwargs)


class segment_sum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(segment_sum, self).__init__(tf.segment_sum, *args, **kwargs)


class self_adjoint_eig(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(self_adjoint_eig, self).__init__(tf.self_adjoint_eig, *args, **kwargs)


class sign(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sign, self).__init__(tf.sign, *args, **kwargs)


class sin(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sin, self).__init__(tf.sin, *args, **kwargs)


class sparse_segment_mean(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_segment_mean, self).__init__(tf.sparse_segment_mean, *args, **kwargs)


class sparse_segment_sqrt_n(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_segment_sqrt_n, self).__init__(tf.sparse_segment_sqrt_n, *args, **kwargs)


class sparse_segment_sqrt_n_grad(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_segment_sqrt_n_grad, self).__init__(tf.sparse_segment_sqrt_n_grad, *args, **kwargs)


class sparse_segment_sum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_segment_sum, self).__init__(tf.sparse_segment_sum, *args, **kwargs)


class sqrt(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sqrt, self).__init__(tf.sqrt, *args, **kwargs)


class square(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(square, self).__init__(tf.square, *args, **kwargs)


class squared_difference(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(squared_difference, self).__init__(tf.squared_difference, *args, **kwargs)


class sub(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sub, self).__init__(tf.sub, *args, **kwargs)


class tan(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(tan, self).__init__(tf.tan, *args, **kwargs)


class trace(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(trace, self).__init__(tf.trace, *args, **kwargs)


class transpose(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(transpose, self).__init__(tf.transpose, *args, **kwargs)


class truediv(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(truediv, self).__init__(tf.truediv, *args, **kwargs)


class unique(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(unique, self).__init__(tf.unique, *args, **kwargs)


class unsorted_segment_sum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(unsorted_segment_sum, self).__init__(tf.unsorted_segment_sum, *args, **kwargs)


class where(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(where, self).__init__(tf.where, *args, **kwargs)


class zeta(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(zeta, self).__init__(tf.zeta, *args, **kwargs)


class as_string(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(as_string, self).__init__(tf.as_string, *args, **kwargs)


class reduce_join(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_join, self).__init__(tf.reduce_join, *args, **kwargs)


class string_join(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(string_join, self).__init__(tf.string_join, *args, **kwargs)


class string_to_hash_bucket(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(string_to_hash_bucket, self).__init__(tf.string_to_hash_bucket, *args, **kwargs)


class string_to_hash_bucket_fast(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(string_to_hash_bucket_fast, self).__init__(tf.string_to_hash_bucket_fast, *args, **kwargs)


class string_to_hash_bucket_strong(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(string_to_hash_bucket_strong, self).__init__(tf.string_to_hash_bucket_strong, *args, **kwargs)


class histogram_fixed_width(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(histogram_fixed_width, self).__init__(tf.histogram_fixed_width, *args, **kwargs)


class add_check_numerics_ops(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_check_numerics_ops, self).__init__(tf.add_check_numerics_ops, *args, **kwargs)


class Assert(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Assert, self).__init__(tf.Assert, *args, **kwargs)


class case(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(case, self).__init__(tf.case, *args, **kwargs)


class check_numerics(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(check_numerics, self).__init__(tf.check_numerics, *args, **kwargs)


class cond(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cond, self).__init__(tf.cond, *args, **kwargs)


class count_up_to(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(count_up_to, self).__init__(tf.count_up_to, *args, **kwargs)


class equal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(equal, self).__init__(tf.equal, *args, **kwargs)


class greater(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(greater, self).__init__(tf.greater, *args, **kwargs)


class greater_equal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(greater_equal, self).__init__(tf.greater_equal, *args, **kwargs)


class group(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(group, self).__init__(tf.group, *args, **kwargs)


class identity(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(identity, self).__init__(tf.identity, *args, **kwargs)


class is_finite(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_finite, self).__init__(tf.is_finite, *args, **kwargs)


class is_inf(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_inf, self).__init__(tf.is_inf, *args, **kwargs)


class is_nan(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_nan, self).__init__(tf.is_nan, *args, **kwargs)


class less(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(less, self).__init__(tf.less, *args, **kwargs)


class less_equal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(less_equal, self).__init__(tf.less_equal, *args, **kwargs)


class logical_and(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(logical_and, self).__init__(tf.logical_and, *args, **kwargs)


class logical_not(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(logical_not, self).__init__(tf.logical_not, *args, **kwargs)


class logical_or(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(logical_or, self).__init__(tf.logical_or, *args, **kwargs)


class logical_xor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(logical_xor, self).__init__(tf.logical_xor, *args, **kwargs)


class no_op(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(no_op, self).__init__(tf.no_op, *args, **kwargs)


class not_equal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(not_equal, self).__init__(tf.not_equal, *args, **kwargs)


class Print(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Print, self).__init__(tf.Print, *args, **kwargs)


class select(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(select, self).__init__(tf.select, *args, **kwargs)


class tuple(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(tuple, self).__init__(tf.tuple, *args, **kwargs)


class verify_tensor_all_finite(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(verify_tensor_all_finite, self).__init__(tf.verify_tensor_all_finite, *args, **kwargs)


class where(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(where, self).__init__(tf.where, *args, **kwargs)


class while_loop(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(while_loop, self).__init__(tf.while_loop, *args, **kwargs)


class foldl(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(foldl, self).__init__(tf.foldl, *args, **kwargs)


class foldr(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(foldr, self).__init__(tf.foldr, *args, **kwargs)


class map_fn(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(map_fn, self).__init__(tf.map_fn, *args, **kwargs)


class scan(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(scan, self).__init__(tf.scan, *args, **kwargs)


class concat(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(concat, self).__init__(tf.concat, *args, **kwargs)


class pack(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(pack, self).__init__(tf.pack, *args, **kwargs)


class split(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(split, self).__init__(tf.split, *args, **kwargs)


class TensorArray(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorArray, self).__init__(tf.TensorArray, *args, **kwargs)


class unpack(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(unpack, self).__init__(tf.unpack, *args, **kwargs)


class delete_session_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(delete_session_tensor, self).__init__(tf.delete_session_tensor, *args, **kwargs)


class get_session_handle(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_session_handle, self).__init__(tf.get_session_handle, *args, **kwargs)


class get_session_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_session_tensor, self).__init__(tf.get_session_tensor, *args, **kwargs)


class adjust_brightness(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(adjust_brightness, self).__init__(tf.adjust_brightness, *args, **kwargs)


class adjust_contrast(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(adjust_contrast, self).__init__(tf.adjust_contrast, *args, **kwargs)


class adjust_hue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(adjust_hue, self).__init__(tf.adjust_hue, *args, **kwargs)


class adjust_saturation(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(adjust_saturation, self).__init__(tf.adjust_saturation, *args, **kwargs)


class central_crop(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(central_crop, self).__init__(tf.central_crop, *args, **kwargs)


class convert_image_dtype(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convert_image_dtype, self).__init__(tf.convert_image_dtype, *args, **kwargs)


class crop_and_resize(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(crop_and_resize, self).__init__(tf.crop_and_resize, *args, **kwargs)


class crop_to_bounding_box(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(crop_to_bounding_box, self).__init__(tf.crop_to_bounding_box, *args, **kwargs)


class decode_jpeg(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(decode_jpeg, self).__init__(tf.decode_jpeg, *args, **kwargs)


class decode_png(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(decode_png, self).__init__(tf.decode_png, *args, **kwargs)


class draw_bounding_boxes(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(draw_bounding_boxes, self).__init__(tf.draw_bounding_boxes, *args, **kwargs)


class encode_jpeg(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(encode_jpeg, self).__init__(tf.encode_jpeg, *args, **kwargs)


class encode_png(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(encode_png, self).__init__(tf.encode_png, *args, **kwargs)


class extract_glimpse(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_glimpse, self).__init__(tf.extract_glimpse, *args, **kwargs)


class flip_left_right(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(flip_left_right, self).__init__(tf.flip_left_right, *args, **kwargs)


class flip_up_down(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(flip_up_down, self).__init__(tf.flip_up_down, *args, **kwargs)


class grayscale_to_rgb(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(grayscale_to_rgb, self).__init__(tf.grayscale_to_rgb, *args, **kwargs)


class hsv_to_rgb(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(hsv_to_rgb, self).__init__(tf.hsv_to_rgb, *args, **kwargs)


class non_max_suppression(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(non_max_suppression, self).__init__(tf.non_max_suppression, *args, **kwargs)


class pad_to_bounding_box(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(pad_to_bounding_box, self).__init__(tf.pad_to_bounding_box, *args, **kwargs)


class per_image_whitening(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(per_image_whitening, self).__init__(tf.per_image_whitening, *args, **kwargs)


class random_brightness(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_brightness, self).__init__(tf.random_brightness, *args, **kwargs)


class random_contrast(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_contrast, self).__init__(tf.random_contrast, *args, **kwargs)


class random_flip_left_right(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_flip_left_right, self).__init__(tf.random_flip_left_right, *args, **kwargs)


class random_flip_up_down(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_flip_up_down, self).__init__(tf.random_flip_up_down, *args, **kwargs)


class random_hue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_hue, self).__init__(tf.random_hue, *args, **kwargs)


class random_saturation(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(random_saturation, self).__init__(tf.random_saturation, *args, **kwargs)


class resize_area(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(resize_area, self).__init__(tf.resize_area, *args, **kwargs)


class resize_bicubic(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(resize_bicubic, self).__init__(tf.resize_bicubic, *args, **kwargs)


class resize_bilinear(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(resize_bilinear, self).__init__(tf.resize_bilinear, *args, **kwargs)


class resize_image_with_crop_or_pad(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(resize_image_with_crop_or_pad, self).__init__(tf.resize_image_with_crop_or_pad, *args, **kwargs)


class resize_images(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(resize_images, self).__init__(tf.resize_images, *args, **kwargs)


class resize_nearest_neighbor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(resize_nearest_neighbor, self).__init__(tf.resize_nearest_neighbor, *args, **kwargs)


class rgb_to_grayscale(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(rgb_to_grayscale, self).__init__(tf.rgb_to_grayscale, *args, **kwargs)


class rgb_to_hsv(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(rgb_to_hsv, self).__init__(tf.rgb_to_hsv, *args, **kwargs)


class rot90(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(rot90, self).__init__(tf.rot90, *args, **kwargs)


class sample_distorted_bounding_box(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sample_distorted_bounding_box, self).__init__(tf.sample_distorted_bounding_box, *args, **kwargs)


class transpose_image(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(transpose_image, self).__init__(tf.transpose_image, *args, **kwargs)


class shape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(shape, self).__init__(tf.shape, *args, **kwargs)


class sparse_add(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_add, self).__init__(tf.sparse_add, *args, **kwargs)


class sparse_concat(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_concat, self).__init__(tf.sparse_concat, *args, **kwargs)


class sparse_fill_empty_rows(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_fill_empty_rows, self).__init__(tf.sparse_fill_empty_rows, *args, **kwargs)


class sparse_maximum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_maximum, self).__init__(tf.sparse_maximum, *args, **kwargs)


class sparse_merge(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_merge, self).__init__(tf.sparse_merge, *args, **kwargs)


class sparse_minimum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_minimum, self).__init__(tf.sparse_minimum, *args, **kwargs)


class sparse_reduce_sum(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_reduce_sum, self).__init__(tf.sparse_reduce_sum, *args, **kwargs)


class sparse_reorder(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_reorder, self).__init__(tf.sparse_reorder, *args, **kwargs)


class sparse_reset_shape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_reset_shape, self).__init__(tf.sparse_reset_shape, *args, **kwargs)


class sparse_reshape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_reshape, self).__init__(tf.sparse_reshape, *args, **kwargs)


class sparse_retain(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_retain, self).__init__(tf.sparse_retain, *args, **kwargs)


class sparse_softmax(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_softmax, self).__init__(tf.sparse_softmax, *args, **kwargs)


class sparse_split(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_split, self).__init__(tf.sparse_split, *args, **kwargs)


class sparse_tensor_dense_matmul(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_tensor_dense_matmul, self).__init__(tf.sparse_tensor_dense_matmul, *args, **kwargs)


class sparse_tensor_to_dense(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_tensor_to_dense, self).__init__(tf.sparse_tensor_to_dense, *args, **kwargs)


class sparse_to_dense(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_to_dense, self).__init__(tf.sparse_to_dense, *args, **kwargs)


class sparse_to_indicator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_to_indicator, self).__init__(tf.sparse_to_indicator, *args, **kwargs)


class SparseTensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SparseTensor, self).__init__(tf.SparseTensor, *args, **kwargs)


class SparseTensorValue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SparseTensorValue, self).__init__(tf.SparseTensorValue, *args, **kwargs)


class batch(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch, self).__init__(tf.batch, *args, **kwargs)


class batch_join(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_join, self).__init__(tf.batch_join, *args, **kwargs)


class decode_csv(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(decode_csv, self).__init__(tf.decode_csv, *args, **kwargs)


class decode_json_example(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(decode_json_example, self).__init__(tf.decode_json_example, *args, **kwargs)


class decode_raw(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(decode_raw, self).__init__(tf.decode_raw, *args, **kwargs)


class FIFOQueue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(FIFOQueue, self).__init__(tf.FIFOQueue, *args, **kwargs)


class FixedLenFeature(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(FixedLenFeature, self).__init__(tf.FixedLenFeature, *args, **kwargs)


class FixedLengthRecordReader(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(FixedLengthRecordReader, self).__init__(tf.FixedLengthRecordReader, *args, **kwargs)


class FixedLenSequenceFeature(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(FixedLenSequenceFeature, self).__init__(tf.FixedLenSequenceFeature, *args, **kwargs)


class IdentityReader(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(IdentityReader, self).__init__(tf.IdentityReader, *args, **kwargs)


class input_producer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(input_producer, self).__init__(tf.input_producer, *args, **kwargs)


class limit_epochs(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(limit_epochs, self).__init__(tf.limit_epochs, *args, **kwargs)


class match_filenames_once(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(match_filenames_once, self).__init__(tf.match_filenames_once, *args, **kwargs)


class matching_files(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(matching_files, self).__init__(tf.matching_files, *args, **kwargs)


class PaddingFIFOQueue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(PaddingFIFOQueue, self).__init__(tf.PaddingFIFOQueue, *args, **kwargs)


class parse_example(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(parse_example, self).__init__(tf.parse_example, *args, **kwargs)


class parse_single_example(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(parse_single_example, self).__init__(tf.parse_single_example, *args, **kwargs)


class placeholder(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(placeholder, self).__init__(tf.placeholder, *args, **kwargs)


class placeholder_with_default(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(placeholder_with_default, self).__init__(tf.placeholder_with_default, *args, **kwargs)


class QueueBase(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(QueueBase, self).__init__(tf.QueueBase, *args, **kwargs)


class RandomShuffleQueue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RandomShuffleQueue, self).__init__(tf.RandomShuffleQueue, *args, **kwargs)


class range_input_producer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(range_input_producer, self).__init__(tf.range_input_producer, *args, **kwargs)


class read_file(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(read_file, self).__init__(tf.read_file, *args, **kwargs)


class ReaderBase(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ReaderBase, self).__init__(tf.ReaderBase, *args, **kwargs)


class shuffle_batch(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(shuffle_batch, self).__init__(tf.shuffle_batch, *args, **kwargs)


class shuffle_batch_join(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(shuffle_batch_join, self).__init__(tf.shuffle_batch_join, *args, **kwargs)


class size(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(size, self).__init__(tf.size, *args, **kwargs)


class slice_input_producer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(slice_input_producer, self).__init__(tf.slice_input_producer, *args, **kwargs)


class sparse_placeholder(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_placeholder, self).__init__(tf.sparse_placeholder, *args, **kwargs)


class string_input_producer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(string_input_producer, self).__init__(tf.string_input_producer, *args, **kwargs)


class TextLineReader(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TextLineReader, self).__init__(tf.TextLineReader, *args, **kwargs)


class TFRecordReader(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TFRecordReader, self).__init__(tf.TFRecordReader, *args, **kwargs)


class VarLenFeature(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(VarLenFeature, self).__init__(tf.VarLenFeature, *args, **kwargs)


class WholeFileReader(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(WholeFileReader, self).__init__(tf.WholeFileReader, *args, **kwargs)


class tf_record_iterator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(tf_record_iterator, self).__init__(tf.tf_record_iterator, *args, **kwargs)


class TFRecordWriter(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TFRecordWriter, self).__init__(tf.TFRecordWriter, *args, **kwargs)


class atrous_conv2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(atrous_conv2d, self).__init__(tf.atrous_conv2d, *args, **kwargs)


class avg_pool(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(avg_pool, self).__init__(tf.avg_pool, *args, **kwargs)


class avg_pool3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(avg_pool3d, self).__init__(tf.avg_pool3d, *args, **kwargs)


class batch_normalization(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_normalization, self).__init__(tf.batch_normalization, *args, **kwargs)


class bias_add(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(bias_add, self).__init__(tf.bias_add, *args, **kwargs)


class bidirectional_rnn(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(bidirectional_rnn, self).__init__(tf.bidirectional_rnn, *args, **kwargs)


class compute_accidental_hits(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(compute_accidental_hits, self).__init__(tf.compute_accidental_hits, *args, **kwargs)


class conv2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(conv2d, self).__init__(tf.conv2d, *args, **kwargs)


class conv2d_transpose(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(conv2d_transpose, self).__init__(tf.conv2d_transpose, *args, **kwargs)


class conv3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(conv3d, self).__init__(tf.conv3d, *args, **kwargs)


class ctc_beam_search_decoder(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ctc_beam_search_decoder, self).__init__(tf.ctc_beam_search_decoder, *args, **kwargs)


class ctc_greedy_decoder(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ctc_greedy_decoder, self).__init__(tf.ctc_greedy_decoder, *args, **kwargs)


class ctc_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ctc_loss, self).__init__(tf.ctc_loss, *args, **kwargs)


class depthwise_conv2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(depthwise_conv2d, self).__init__(tf.depthwise_conv2d, *args, **kwargs)


class depthwise_conv2d_native(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(depthwise_conv2d_native, self).__init__(tf.depthwise_conv2d_native, *args, **kwargs)


class dilation2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(dilation2d, self).__init__(tf.dilation2d, *args, **kwargs)


class dropout(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(dropout, self).__init__(tf.dropout, *args, **kwargs)


class dynamic_rnn(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(dynamic_rnn, self).__init__(tf.dynamic_rnn, *args, **kwargs)


class elu(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(elu, self).__init__(tf.elu, *args, **kwargs)


class embedding_lookup(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(embedding_lookup, self).__init__(tf.embedding_lookup, *args, **kwargs)


class embedding_lookup_sparse(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(embedding_lookup_sparse, self).__init__(tf.embedding_lookup_sparse, *args, **kwargs)


class erosion2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(erosion2d, self).__init__(tf.erosion2d, *args, **kwargs)


class fixed_unigram_candidate_sampler(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(fixed_unigram_candidate_sampler, self).__init__(tf.fixed_unigram_candidate_sampler, *args, **kwargs)


class in_top_k(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(in_top_k, self).__init__(tf.in_top_k, *args, **kwargs)


class l2_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(l2_loss, self).__init__(tf.l2_loss, *args, **kwargs)


class l2_normalize(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(l2_normalize, self).__init__(tf.l2_normalize, *args, **kwargs)


class learned_unigram_candidate_sampler(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(learned_unigram_candidate_sampler, self).__init__(tf.learned_unigram_candidate_sampler, *args, **kwargs)


class local_response_normalization(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(local_response_normalization, self).__init__(tf.local_response_normalization, *args, **kwargs)


class log_softmax(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(log_softmax, self).__init__(tf.log_softmax, *args, **kwargs)


class log_uniform_candidate_sampler(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(log_uniform_candidate_sampler, self).__init__(tf.log_uniform_candidate_sampler, *args, **kwargs)


class max_pool(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(max_pool, self).__init__(tf.max_pool, *args, **kwargs)


class max_pool3d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(max_pool3d, self).__init__(tf.max_pool3d, *args, **kwargs)


class max_pool_with_argmax(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(max_pool_with_argmax, self).__init__(tf.max_pool_with_argmax, *args, **kwargs)


class moments(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(moments, self).__init__(tf.moments, *args, **kwargs)


class nce_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(nce_loss, self).__init__(tf.nce_loss, *args, **kwargs)


class normalize_moments(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(normalize_moments, self).__init__(tf.normalize_moments, *args, **kwargs)


class relu(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(relu, self).__init__(tf.relu, *args, **kwargs)


class relu6(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(relu6, self).__init__(tf.relu6, *args, **kwargs)


class rnn(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(rnn, self).__init__(tf.rnn, *args, **kwargs)


class sampled_softmax_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sampled_softmax_loss, self).__init__(tf.sampled_softmax_loss, *args, **kwargs)


class separable_conv2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(separable_conv2d, self).__init__(tf.separable_conv2d, *args, **kwargs)


class sigmoid(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sigmoid, self).__init__(tf.sigmoid, *args, **kwargs)


class sigmoid_cross_entropy_with_logits(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sigmoid_cross_entropy_with_logits, self).__init__(tf.sigmoid_cross_entropy_with_logits, *args, **kwargs)


class softmax(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(softmax, self).__init__(tf.softmax, *args, **kwargs)


class softmax_cross_entropy_with_logits(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(softmax_cross_entropy_with_logits, self).__init__(tf.softmax_cross_entropy_with_logits, *args, **kwargs)


class softplus(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(softplus, self).__init__(tf.softplus, *args, **kwargs)


class softsign(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(softsign, self).__init__(tf.softsign, *args, **kwargs)


class sparse_softmax_cross_entropy_with_logits(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sparse_softmax_cross_entropy_with_logits, self).__init__(tf.sparse_softmax_cross_entropy_with_logits, *args, **kwargs)


class state_saving_rnn(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(state_saving_rnn, self).__init__(tf.state_saving_rnn, *args, **kwargs)


class sufficient_statistics(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sufficient_statistics, self).__init__(tf.sufficient_statistics, *args, **kwargs)


class tanh(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(tanh, self).__init__(tf.tanh, *args, **kwargs)


class top_k(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(top_k, self).__init__(tf.top_k, *args, **kwargs)


class uniform_candidate_sampler(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(uniform_candidate_sampler, self).__init__(tf.uniform_candidate_sampler, *args, **kwargs)


class weighted_cross_entropy_with_logits(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(weighted_cross_entropy_with_logits, self).__init__(tf.weighted_cross_entropy_with_logits, *args, **kwargs)


class BasicLSTMCell(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(BasicLSTMCell, self).__init__(tf.BasicLSTMCell, *args, **kwargs)


class BasicRNNCell(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(BasicRNNCell, self).__init__(tf.BasicRNNCell, *args, **kwargs)


class DropoutWrapper(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DropoutWrapper, self).__init__(tf.DropoutWrapper, *args, **kwargs)


class EmbeddingWrapper(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(EmbeddingWrapper, self).__init__(tf.EmbeddingWrapper, *args, **kwargs)


class GRUCell(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(GRUCell, self).__init__(tf.GRUCell, *args, **kwargs)


class InputProjectionWrapper(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(InputProjectionWrapper, self).__init__(tf.InputProjectionWrapper, *args, **kwargs)


class LSTMCell(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(LSTMCell, self).__init__(tf.LSTMCell, *args, **kwargs)


class LSTMStateTuple(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(LSTMStateTuple, self).__init__(tf.LSTMStateTuple, *args, **kwargs)


class MultiRNNCell(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultiRNNCell, self).__init__(tf.MultiRNNCell, *args, **kwargs)


class OutputProjectionWrapper(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(OutputProjectionWrapper, self).__init__(tf.OutputProjectionWrapper, *args, **kwargs)


class RNNCell(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RNNCell, self).__init__(tf.RNNCell, *args, **kwargs)


class AbortedError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(AbortedError, self).__init__(tf.AbortedError, *args, **kwargs)


class AlreadyExistsError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(AlreadyExistsError, self).__init__(tf.AlreadyExistsError, *args, **kwargs)


class CancelledError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(CancelledError, self).__init__(tf.CancelledError, *args, **kwargs)


class DataLossError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DataLossError, self).__init__(tf.DataLossError, *args, **kwargs)


class DeadlineExceededError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DeadlineExceededError, self).__init__(tf.DeadlineExceededError, *args, **kwargs)


class FailedPreconditionError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(FailedPreconditionError, self).__init__(tf.FailedPreconditionError, *args, **kwargs)


class get_default_session(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_default_session, self).__init__(tf.get_default_session, *args, **kwargs)


class InteractiveSession(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(InteractiveSession, self).__init__(tf.InteractiveSession, *args, **kwargs)


class InternalError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(InternalError, self).__init__(tf.InternalError, *args, **kwargs)


class InvalidArgumentError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(InvalidArgumentError, self).__init__(tf.InvalidArgumentError, *args, **kwargs)


class NotFoundError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(NotFoundError, self).__init__(tf.NotFoundError, *args, **kwargs)


class OpError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(OpError, self).__init__(tf.OpError, *args, **kwargs)


class OutOfRangeError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(OutOfRangeError, self).__init__(tf.OutOfRangeError, *args, **kwargs)


class PermissionDeniedError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(PermissionDeniedError, self).__init__(tf.PermissionDeniedError, *args, **kwargs)


class ResourceExhaustedError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ResourceExhaustedError, self).__init__(tf.ResourceExhaustedError, *args, **kwargs)


class Session(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Session, self).__init__(tf.Session, *args, **kwargs)


class UnauthenticatedError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(UnauthenticatedError, self).__init__(tf.UnauthenticatedError, *args, **kwargs)


class UnavailableError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(UnavailableError, self).__init__(tf.UnavailableError, *args, **kwargs)


class UnimplementedError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(UnimplementedError, self).__init__(tf.UnimplementedError, *args, **kwargs)


class UnknownError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(UnknownError, self).__init__(tf.UnknownError, *args, **kwargs)


class AdadeltaOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(AdadeltaOptimizer, self).__init__(tf.AdadeltaOptimizer, *args, **kwargs)


class AdagradOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(AdagradOptimizer, self).__init__(tf.AdagradOptimizer, *args, **kwargs)


class AdamOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(AdamOptimizer, self).__init__(tf.AdamOptimizer, *args, **kwargs)


class add_queue_runner(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_queue_runner, self).__init__(tf.add_queue_runner, *args, **kwargs)


class AggregationMethod(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(AggregationMethod, self).__init__(tf.AggregationMethod, *args, **kwargs)


class audio_summary(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(audio_summary, self).__init__(tf.audio_summary, *args, **kwargs)


class clip_by_average_norm(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(clip_by_average_norm, self).__init__(tf.clip_by_average_norm, *args, **kwargs)


class clip_by_global_norm(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(clip_by_global_norm, self).__init__(tf.clip_by_global_norm, *args, **kwargs)


class clip_by_norm(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(clip_by_norm, self).__init__(tf.clip_by_norm, *args, **kwargs)


class clip_by_value(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(clip_by_value, self).__init__(tf.clip_by_value, *args, **kwargs)


class ClusterSpec(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ClusterSpec, self).__init__(tf.ClusterSpec, *args, **kwargs)


class Coordinator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Coordinator, self).__init__(tf.Coordinator, *args, **kwargs)


class do_quantize_training_on_graphdef(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(do_quantize_training_on_graphdef, self).__init__(tf.do_quantize_training_on_graphdef, *args, **kwargs)


class exponential_decay(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(exponential_decay, self).__init__(tf.exponential_decay, *args, **kwargs)


class ExponentialMovingAverage(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ExponentialMovingAverage, self).__init__(tf.ExponentialMovingAverage, *args, **kwargs)


class FtrlOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(FtrlOptimizer, self).__init__(tf.FtrlOptimizer, *args, **kwargs)


class generate_checkpoint_state_proto(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(generate_checkpoint_state_proto, self).__init__(tf.generate_checkpoint_state_proto, *args, **kwargs)


class global_norm(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(global_norm, self).__init__(tf.global_norm, *args, **kwargs)


class global_step(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(global_step, self).__init__(tf.global_step, *args, **kwargs)


class GradientDescentOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(GradientDescentOptimizer, self).__init__(tf.GradientDescentOptimizer, *args, **kwargs)


class gradients(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(gradients, self).__init__(tf.gradients, *args, **kwargs)


class histogram_summary(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(histogram_summary, self).__init__(tf.histogram_summary, *args, **kwargs)


class image_summary(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(image_summary, self).__init__(tf.image_summary, *args, **kwargs)


class LooperThread(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(LooperThread, self).__init__(tf.LooperThread, *args, **kwargs)


class merge_all_summaries(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(merge_all_summaries, self).__init__(tf.merge_all_summaries, *args, **kwargs)


class merge_summary(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(merge_summary, self).__init__(tf.merge_summary, *args, **kwargs)


class MomentumOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MomentumOptimizer, self).__init__(tf.MomentumOptimizer, *args, **kwargs)


class Optimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Optimizer, self).__init__(tf.Optimizer, *args, **kwargs)


class QueueRunner(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(QueueRunner, self).__init__(tf.QueueRunner, *args, **kwargs)


class replica_device_setter(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(replica_device_setter, self).__init__(tf.replica_device_setter, *args, **kwargs)


class RMSPropOptimizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RMSPropOptimizer, self).__init__(tf.RMSPropOptimizer, *args, **kwargs)


class scalar_summary(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(scalar_summary, self).__init__(tf.scalar_summary, *args, **kwargs)


class Server(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Server, self).__init__(tf.Server, *args, **kwargs)


class SessionManager(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SessionManager, self).__init__(tf.SessionManager, *args, **kwargs)


class start_queue_runners(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(start_queue_runners, self).__init__(tf.start_queue_runners, *args, **kwargs)


class stop_gradient(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(stop_gradient, self).__init__(tf.stop_gradient, *args, **kwargs)


class summary_iterator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(summary_iterator, self).__init__(tf.summary_iterator, *args, **kwargs)


class SummaryWriter(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SummaryWriter, self).__init__(tf.SummaryWriter, *args, **kwargs)


class Supervisor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Supervisor, self).__init__(tf.Supervisor, *args, **kwargs)


class write_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(write_graph, self).__init__(tf.write_graph, *args, **kwargs)


class zero_fraction(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(zero_fraction, self).__init__(tf.zero_fraction, *args, **kwargs)


class py_func(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(py_func, self).__init__(tf.py_func, *args, **kwargs)


class tensor_summary(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(tensor_summary, self).__init__(tf.tensor_summary, *args, **kwargs)


class assert_equal_graph_def(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_equal_graph_def, self).__init__(tf.assert_equal_graph_def, *args, **kwargs)


class compute_gradient(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(compute_gradient, self).__init__(tf.compute_gradient, *args, **kwargs)


class compute_gradient_error(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(compute_gradient_error, self).__init__(tf.compute_gradient_error, *args, **kwargs)


class get_temp_dir(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_temp_dir, self).__init__(tf.get_temp_dir, *args, **kwargs)


class is_built_with_cuda(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_built_with_cuda, self).__init__(tf.is_built_with_cuda, *args, **kwargs)


class main(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(main, self).__init__(tf.main, *args, **kwargs)


class DistributionTensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DistributionTensor, self).__init__(tf.contreib.bayesflow.stochastic_graph.DistributionTensor, *args, **kwargs)


class get_current_value_type(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_current_value_type, self).__init__(tf.contreib.bayesflow.stochastic_graph.get_current_value_type, *args, **kwargs)


class get_score_function_with_baseline(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_score_function_with_baseline, self).__init__(tf.contreib.bayesflow.stochastic_graph.get_score_function_with_baseline, *args, **kwargs)


class MeanValue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MeanValue, self).__init__(tf.contreib.bayesflow.stochastic_graph.MeanValue, *args, **kwargs)


class NoValueTypeSetError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(NoValueTypeSetError, self).__init__(tf.contreib.bayesflow.stochastic_graph.NoValueTypeSetError, *args, **kwargs)


class SampleAndReshapeValue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SampleAndReshapeValue, self).__init__(tf.contreib.bayesflow.stochastic_graph.SampleAndReshapeValue, *args, **kwargs)


class SampleValue(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SampleValue, self).__init__(tf.contreib.bayesflow.stochastic_graph.SampleValue, *args, **kwargs)


class score_function(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(score_function, self).__init__(tf.contreib.bayesflow.stochastic_graph.score_function, *args, **kwargs)


class StochasticTensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(StochasticTensor, self).__init__(tf.contreib.bayesflow.stochastic_graph.StochasticTensor, *args, **kwargs)


class surrogate_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(surrogate_loss, self).__init__(tf.contreib.bayesflow.stochastic_graph.surrogate_loss, *args, **kwargs)


class value_type(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(value_type, self).__init__(tf.contreib.bayesflow.stochastic_graph.value_type, *args, **kwargs)


class BaseDistribution(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(BaseDistribution, self).__init__(tf.contrib.distributions.BaseDistribution, *args, **kwargs)


class batch_matrix_diag_transform(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_matrix_diag_transform, self).__init__(tf.contrib.distributions.batch_matrix_diag_transform, *args, **kwargs)


class Bernoulli(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Bernoulli, self).__init__(tf.contrib.distributions.Bernoulli, *args, **kwargs)


class Beta(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Beta, self).__init__(tf.contrib.distributions.Beta, *args, **kwargs)


class Categorical(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Categorical, self).__init__(tf.contrib.distributions.Categorical, *args, **kwargs)


class Chi2(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Chi2, self).__init__(tf.contrib.distributions.Chi2, *args, **kwargs)


class Dirichlet(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Dirichlet, self).__init__(tf.contrib.distributions.Dirichlet, *args, **kwargs)


class DirichletMultinomial(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DirichletMultinomial, self).__init__(tf.contrib.distributions.DirichletMultinomial, *args, **kwargs)


class Distribution(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Distribution, self).__init__(tf.contrib.distributions.Distribution, *args, **kwargs)


class Exponential(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Exponential, self).__init__(tf.contrib.distributions.Exponential, *args, **kwargs)


class Gamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Gamma, self).__init__(tf.contrib.distributions.Gamma, *args, **kwargs)


class InverseGamma(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(InverseGamma, self).__init__(tf.contrib.distributions.InverseGamma, *args, **kwargs)


class kl(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(kl, self).__init__(tf.contrib.distributions.kl, *args, **kwargs)


class Laplace(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Laplace, self).__init__(tf.contrib.distributions.Laplace, *args, **kwargs)


class MultivariateNormalCholesky(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultivariateNormalCholesky, self).__init__(tf.contrib.distributions.MultivariateNormalCholesky, *args, **kwargs)


class MultivariateNormalDiag(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultivariateNormalDiag, self).__init__(tf.contrib.distributions.MultivariateNormalDiag, *args, **kwargs)


class MultivariateNormalFull(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(MultivariateNormalFull, self).__init__(tf.contrib.distributions.MultivariateNormalFull, *args, **kwargs)


class Normal(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Normal, self).__init__(tf.contrib.distributions.Normal, *args, **kwargs)


class normal_congugates_known_sigma_predictive(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(normal_congugates_known_sigma_predictive, self).__init__(tf.contrib.distributions.normal_congugates_known_sigma_predictive, *args, **kwargs)


class normal_conjugates_known_sigma_posterior(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(normal_conjugates_known_sigma_posterior, self).__init__(tf.contrib.distributions.normal_conjugates_known_sigma_posterior, *args, **kwargs)


class RegisterKL(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RegisterKL, self).__init__(tf.contrib.distributions.RegisterKL, *args, **kwargs)


class StudentT(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(StudentT, self).__init__(tf.contrib.distributions.StudentT, *args, **kwargs)


class TransformedDistribution(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TransformedDistribution, self).__init__(tf.contrib.distributions.TransformedDistribution, *args, **kwargs)


class Uniform(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Uniform, self).__init__(tf.contrib.distributions.Uniform, *args, **kwargs)


class decode_audio(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(decode_audio, self).__init__(tf.contrib.ffmpeg.decode_audio, *args, **kwargs)


class encode_audio(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(encode_audio, self).__init__(tf.contrib.ffmpeg.encode_audio, *args, **kwargs)


class add_arg_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_arg_scope, self).__init__(tf.contrib.framework.add_arg_scope, *args, **kwargs)


class add_model_variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_model_variable, self).__init__(tf.contrib.framework.add_model_variable, *args, **kwargs)


class arg_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(arg_scope, self).__init__(tf.contrib.framework.arg_scope, *args, **kwargs)


class arg_scoped_arguments(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(arg_scoped_arguments, self).__init__(tf.contrib.framework.arg_scoped_arguments, *args, **kwargs)


class assert_global_step(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_global_step, self).__init__(tf.contrib.framework.assert_global_step, *args, **kwargs)


class assert_or_get_global_step(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_or_get_global_step, self).__init__(tf.contrib.framework.assert_or_get_global_step, *args, **kwargs)


class assert_same_float_dtype(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_same_float_dtype, self).__init__(tf.contrib.framework.assert_same_float_dtype, *args, **kwargs)


class assert_scalar_int(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(assert_scalar_int, self).__init__(tf.contrib.framework.assert_scalar_int, *args, **kwargs)


class convert_to_tensor_or_sparse_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convert_to_tensor_or_sparse_tensor, self).__init__(tf.contrib.framework.convert_to_tensor_or_sparse_tensor, *args, **kwargs)


class create_global_step(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(create_global_step, self).__init__(tf.contrib.framework.create_global_step, *args, **kwargs)


class deprecated(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(deprecated, self).__init__(tf.contrib.framework.deprecated, *args, **kwargs)


class get_global_step(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_global_step, self).__init__(tf.contrib.framework.get_global_step, *args, **kwargs)


class get_graph_from_inputs(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_graph_from_inputs, self).__init__(tf.contrib.framework.get_graph_from_inputs, *args, **kwargs)


class get_local_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_local_variables, self).__init__(tf.contrib.framework.get_local_variables, *args, **kwargs)


class get_model_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_model_variables, self).__init__(tf.contrib.framework.get_model_variables, *args, **kwargs)


class get_or_create_global_step(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_or_create_global_step, self).__init__(tf.contrib.framework.get_or_create_global_step, *args, **kwargs)


class get_unique_variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_unique_variable, self).__init__(tf.contrib.framework.get_unique_variable, *args, **kwargs)


class get_variables(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_variables, self).__init__(tf.contrib.framework.get_variables, *args, **kwargs)


class get_variables_by_name(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_variables_by_name, self).__init__(tf.contrib.framework.get_variables_by_name, *args, **kwargs)


class get_variables_by_suffix(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_variables_by_suffix, self).__init__(tf.contrib.framework.get_variables_by_suffix, *args, **kwargs)


class get_variables_to_restore(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_variables_to_restore, self).__init__(tf.contrib.framework.get_variables_to_restore, *args, **kwargs)


class has_arg_scope(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(has_arg_scope, self).__init__(tf.contrib.framework.has_arg_scope, *args, **kwargs)


class is_non_decreasing(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_non_decreasing, self).__init__(tf.contrib.framework.is_non_decreasing, *args, **kwargs)


class is_numeric_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_numeric_tensor, self).__init__(tf.contrib.framework.is_numeric_tensor, *args, **kwargs)


class is_strictly_increasing(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_strictly_increasing, self).__init__(tf.contrib.framework.is_strictly_increasing, *args, **kwargs)


class is_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(is_tensor, self).__init__(tf.contrib.framework.is_tensor, *args, **kwargs)


class local_variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(local_variable, self).__init__(tf.contrib.framework.local_variable, *args, **kwargs)


class model_variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(model_variable, self).__init__(tf.contrib.framework.model_variable, *args, **kwargs)


class reduce_sum_n(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(reduce_sum_n, self).__init__(tf.contrib.framework.reduce_sum_n, *args, **kwargs)


class safe_embedding_lookup_sparse(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(safe_embedding_lookup_sparse, self).__init__(tf.contrib.framework.safe_embedding_lookup_sparse, *args, **kwargs)


class variable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(variable, self).__init__(tf.contrib.framework.variable, *args, **kwargs)


class VariableDeviceChooser(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(VariableDeviceChooser, self).__init__(tf.contrib.framework.VariableDeviceChooser, *args, **kwargs)


class with_same_shape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(with_same_shape, self).__init__(tf.contrib.framework.with_same_shape, *args, **kwargs)


class with_shape(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(with_shape, self).__init__(tf.contrib.framework.with_shape, *args, **kwargs)


class apply_regularization(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(apply_regularization, self).__init__(tf.contrib.layers.apply_regularization, *args, **kwargs)


class avg_pool2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(avg_pool2d, self).__init__(tf.contrib.layers.avg_pool2d, *args, **kwargs)


class batch_norm(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(batch_norm, self).__init__(tf.contrib.layers.batch_norm, *args, **kwargs)


class convolution2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convolution2d, self).__init__(tf.contrib.layers.convolution2d, *args, **kwargs)


class convolution2d_in_plane(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convolution2d_in_plane, self).__init__(tf.contrib.layers.convolution2d_in_plane, *args, **kwargs)


class convolution2d_transpose(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(convolution2d_transpose, self).__init__(tf.contrib.layers.convolution2d_transpose, *args, **kwargs)


class flatten(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(flatten, self).__init__(tf.contrib.layers.flatten, *args, **kwargs)


class fully_connected(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(fully_connected, self).__init__(tf.contrib.layers.fully_connected, *args, **kwargs)


class l1_regularizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(l1_regularizer, self).__init__(tf.contrib.layers.l1_regularizer, *args, **kwargs)


class l2_regularizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(l2_regularizer, self).__init__(tf.contrib.layers.l2_regularizer, *args, **kwargs)


class max_pool2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(max_pool2d, self).__init__(tf.contrib.layers.max_pool2d, *args, **kwargs)


class one_hot_encoding(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(one_hot_encoding, self).__init__(tf.contrib.layers.one_hot_encoding, *args, **kwargs)


class optimize_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(optimize_loss, self).__init__(tf.contrib.layers.optimize_loss, *args, **kwargs)


class repeat(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(repeat, self).__init__(tf.contrib.layers.repeat, *args, **kwargs)


class separable_convolution2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(separable_convolution2d, self).__init__(tf.contrib.layers.separable_convolution2d, *args, **kwargs)


class stack(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(stack, self).__init__(tf.contrib.layers.stack, *args, **kwargs)


class sum_regularizer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sum_regularizer, self).__init__(tf.contrib.layers.sum_regularizer, *args, **kwargs)


class summarize_activation(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(summarize_activation, self).__init__(tf.contrib.layers.summarize_activation, *args, **kwargs)


class summarize_activations(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(summarize_activations, self).__init__(tf.contrib.layers.summarize_activations, *args, **kwargs)


class summarize_collection(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(summarize_collection, self).__init__(tf.contrib.layers.summarize_collection, *args, **kwargs)


class summarize_tensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(summarize_tensor, self).__init__(tf.contrib.layers.summarize_tensor, *args, **kwargs)


class summarize_tensors(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(summarize_tensors, self).__init__(tf.contrib.layers.summarize_tensors, *args, **kwargs)


class unit_norm(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(unit_norm, self).__init__(tf.contrib.layers.unit_norm, *args, **kwargs)


class variance_scaling_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(variance_scaling_initializer, self).__init__(tf.contrib.layers.variance_scaling_initializer, *args, **kwargs)


class xavier_initializer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(xavier_initializer, self).__init__(tf.contrib.layers.xavier_initializer, *args, **kwargs)


class xavier_initializer_conv2d(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(xavier_initializer_conv2d, self).__init__(tf.contrib.layers.xavier_initializer_conv2d, *args, **kwargs)


class BaseEstimator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(BaseEstimator, self).__init__(tf.contrib.learn.BaseEstimator, *args, **kwargs)


class DNNClassifier(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DNNClassifier, self).__init__(tf.contrib.learn.DNNClassifier, *args, **kwargs)


class DNNRegressor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(DNNRegressor, self).__init__(tf.contrib.learn.DNNRegressor, *args, **kwargs)


class Estimator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(Estimator, self).__init__(tf.contrib.learn.Estimator, *args, **kwargs)


class evaluate(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(evaluate, self).__init__(tf.contrib.learn.evaluate, *args, **kwargs)


class extract_dask_data(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_dask_data, self).__init__(tf.contrib.learn.extract_dask_data, *args, **kwargs)


class extract_dask_labels(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_dask_labels, self).__init__(tf.contrib.learn.extract_dask_labels, *args, **kwargs)


class extract_pandas_data(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_pandas_data, self).__init__(tf.contrib.learn.extract_pandas_data, *args, **kwargs)


class extract_pandas_labels(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_pandas_labels, self).__init__(tf.contrib.learn.extract_pandas_labels, *args, **kwargs)


class extract_pandas_matrix(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(extract_pandas_matrix, self).__init__(tf.contrib.learn.extract_pandas_matrix, *args, **kwargs)


class infer(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(infer, self).__init__(tf.contrib.learn.infer, *args, **kwargs)


class LinearClassifier(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(LinearClassifier, self).__init__(tf.contrib.learn.LinearClassifier, *args, **kwargs)


class LinearRegressor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(LinearRegressor, self).__init__(tf.contrib.learn.LinearRegressor, *args, **kwargs)


class ModeKeys(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ModeKeys, self).__init__(tf.contrib.learn.ModeKeys, *args, **kwargs)


class NanLossDuringTrainingError(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(NanLossDuringTrainingError, self).__init__(tf.contrib.learn.NanLossDuringTrainingError, *args, **kwargs)


class read_batch_examples(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(read_batch_examples, self).__init__(tf.contrib.learn.read_batch_examples, *args, **kwargs)


class read_batch_features(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(read_batch_features, self).__init__(tf.contrib.learn.read_batch_features, *args, **kwargs)


class read_batch_record_features(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(read_batch_record_features, self).__init__(tf.contrib.learn.read_batch_record_features, *args, **kwargs)


class run_feeds(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(run_feeds, self).__init__(tf.contrib.learn.run_feeds, *args, **kwargs)


class run_n(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(run_n, self).__init__(tf.contrib.learn.run_n, *args, **kwargs)


class RunConfig(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(RunConfig, self).__init__(tf.contrib.learn.RunConfig, *args, **kwargs)


class TensorFlowClassifier(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowClassifier, self).__init__(tf.contrib.learn.TensorFlowClassifier, *args, **kwargs)


class TensorFlowDNNClassifier(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowDNNClassifier, self).__init__(tf.contrib.learn.TensorFlowDNNClassifier, *args, **kwargs)


class TensorFlowDNNRegressor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowDNNRegressor, self).__init__(tf.contrib.learn.TensorFlowDNNRegressor, *args, **kwargs)


class TensorFlowEstimator(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowEstimator, self).__init__(tf.contrib.learn.TensorFlowEstimator, *args, **kwargs)


class TensorFlowLinearClassifier(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowLinearClassifier, self).__init__(tf.contrib.learn.TensorFlowLinearClassifier, *args, **kwargs)


class TensorFlowLinearRegressor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowLinearRegressor, self).__init__(tf.contrib.learn.TensorFlowLinearRegressor, *args, **kwargs)


class TensorFlowRegressor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowRegressor, self).__init__(tf.contrib.learn.TensorFlowRegressor, *args, **kwargs)


class TensorFlowRNNClassifier(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowRNNClassifier, self).__init__(tf.contrib.learn.TensorFlowRNNClassifier, *args, **kwargs)


class TensorFlowRNNRegressor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(TensorFlowRNNRegressor, self).__init__(tf.contrib.learn.TensorFlowRNNRegressor, *args, **kwargs)


class train(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(train, self).__init__(tf.contrib.learn.train, *args, **kwargs)


class BaseMonitor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(BaseMonitor, self).__init__(tf.contrib.monitors.BaseMonitor, *args, **kwargs)


class CaptureVariable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(CaptureVariable, self).__init__(tf.contrib.monitors.CaptureVariable, *args, **kwargs)


class CheckpointSaver(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(CheckpointSaver, self).__init__(tf.contrib.monitors.CheckpointSaver, *args, **kwargs)


class EveryN(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(EveryN, self).__init__(tf.contrib.monitors.EveryN, *args, **kwargs)


class ExportMonitor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ExportMonitor, self).__init__(tf.contrib.monitors.ExportMonitor, *args, **kwargs)


class get_default_monitors(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_default_monitors, self).__init__(tf.contrib.monitors.get_default_monitors, *args, **kwargs)


class GraphDump(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(GraphDump, self).__init__(tf.contrib.monitors.GraphDump, *args, **kwargs)


class LoggingTrainable(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(LoggingTrainable, self).__init__(tf.contrib.monitors.LoggingTrainable, *args, **kwargs)


class NanLoss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(NanLoss, self).__init__(tf.contrib.monitors.NanLoss, *args, **kwargs)


class PrintTensor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(PrintTensor, self).__init__(tf.contrib.monitors.PrintTensor, *args, **kwargs)


class StepCounter(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(StepCounter, self).__init__(tf.contrib.monitors.StepCounter, *args, **kwargs)


class StopAtStep(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(StopAtStep, self).__init__(tf.contrib.monitors.StopAtStep, *args, **kwargs)


class SummarySaver(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SummarySaver, self).__init__(tf.contrib.monitors.SummarySaver, *args, **kwargs)


class SummaryWriterCache(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(SummaryWriterCache, self).__init__(tf.contrib.monitors.SummaryWriterCache, *args, **kwargs)


class ValidationMonitor(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ValidationMonitor, self).__init__(tf.contrib.monitors.ValidationMonitor, *args, **kwargs)


class absolute_difference(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(absolute_difference, self).__init__(tf.contrib.losses.absolute_difference, *args, **kwargs)


class add_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(add_loss, self).__init__(tf.contrib.losses.add_loss, *args, **kwargs)


class cosine_distance(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(cosine_distance, self).__init__(tf.contrib.losses.cosine_distance, *args, **kwargs)


class get_losses(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_losses, self).__init__(tf.contrib.losses.get_losses, *args, **kwargs)


class get_regularization_losses(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_regularization_losses, self).__init__(tf.contrib.losses.get_regularization_losses, *args, **kwargs)


class get_total_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_total_loss, self).__init__(tf.contrib.losses.get_total_loss, *args, **kwargs)


class log_loss(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(log_loss, self).__init__(tf.contrib.losses.log_loss, *args, **kwargs)


class sigmoid_cross_entropy(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sigmoid_cross_entropy, self).__init__(tf.contrib.losses.sigmoid_cross_entropy, *args, **kwargs)


class softmax_cross_entropy(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(softmax_cross_entropy, self).__init__(tf.contrib.losses.softmax_cross_entropy, *args, **kwargs)


class sum_of_pairwise_squares(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sum_of_pairwise_squares, self).__init__(tf.contrib.losses.sum_of_pairwise_squares, *args, **kwargs)


class sum_of_squares(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(sum_of_squares, self).__init__(tf.contrib.losses.sum_of_squares, *args, **kwargs)


class accuracy(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(accuracy, self).__init__(tf.contrib.metrics.accuracy, *args, **kwargs)


class aggregate_metric_map(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(aggregate_metric_map, self).__init__(tf.contrib.metrics.aggregate_metric_map, *args, **kwargs)


class aggregate_metrics(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(aggregate_metrics, self).__init__(tf.contrib.metrics.aggregate_metrics, *args, **kwargs)


class auc_using_histogram(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(auc_using_histogram, self).__init__(tf.contrib.metrics.auc_using_histogram, *args, **kwargs)


class confusion_matrix(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(confusion_matrix, self).__init__(tf.contrib.metrics.confusion_matrix, *args, **kwargs)


class set_difference(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(set_difference, self).__init__(tf.contrib.metrics.set_difference, *args, **kwargs)


class set_intersection(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(set_intersection, self).__init__(tf.contrib.metrics.set_intersection, *args, **kwargs)


class set_size(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(set_size, self).__init__(tf.contrib.metrics.set_size, *args, **kwargs)


class set_union(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(set_union, self).__init__(tf.contrib.metrics.set_union, *args, **kwargs)


class streaming_accuracy(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_accuracy, self).__init__(tf.contrib.metrics.streaming_accuracy, *args, **kwargs)


class streaming_auc(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_auc, self).__init__(tf.contrib.metrics.streaming_auc, *args, **kwargs)


class streaming_mean(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_mean, self).__init__(tf.contrib.metrics.streaming_mean, *args, **kwargs)


class streaming_mean_absolute_error(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_mean_absolute_error, self).__init__(tf.contrib.metrics.streaming_mean_absolute_error, *args, **kwargs)


class streaming_mean_cosine_distance(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_mean_cosine_distance, self).__init__(tf.contrib.metrics.streaming_mean_cosine_distance, *args, **kwargs)


class streaming_mean_iou(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_mean_iou, self).__init__(tf.contrib.metrics.streaming_mean_iou, *args, **kwargs)


class streaming_mean_relative_error(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_mean_relative_error, self).__init__(tf.contrib.metrics.streaming_mean_relative_error, *args, **kwargs)


class streaming_mean_squared_error(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_mean_squared_error, self).__init__(tf.contrib.metrics.streaming_mean_squared_error, *args, **kwargs)


class streaming_percentage_less(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_percentage_less, self).__init__(tf.contrib.metrics.streaming_percentage_less, *args, **kwargs)


class streaming_precision(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_precision, self).__init__(tf.contrib.metrics.streaming_precision, *args, **kwargs)


class streaming_recall(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_recall, self).__init__(tf.contrib.metrics.streaming_recall, *args, **kwargs)


class streaming_recall_at_k(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_recall_at_k, self).__init__(tf.contrib.metrics.streaming_recall_at_k, *args, **kwargs)


class streaming_root_mean_squared_error(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_root_mean_squared_error, self).__init__(tf.contrib.metrics.streaming_root_mean_squared_error, *args, **kwargs)


class streaming_sparse_precision_at_k(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_sparse_precision_at_k, self).__init__(tf.contrib.metrics.streaming_sparse_precision_at_k, *args, **kwargs)


class streaming_sparse_recall_at_k(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(streaming_sparse_recall_at_k, self).__init__(tf.contrib.metrics.streaming_sparse_recall_at_k, *args, **kwargs)


class constant_value(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(constant_value, self).__init__(tf.contrib.util.constant_value, *args, **kwargs)


class make_ndarray(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(make_ndarray, self).__init__(tf.contrib.util.make_ndarray, *args, **kwargs)


class make_tensor_proto(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(make_tensor_proto, self).__init__(tf.contrib.util.make_tensor_proto, *args, **kwargs)


class ops_used_by_graph_def(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(ops_used_by_graph_def, self).__init__(tf.contrib.util.ops_used_by_graph_def, *args, **kwargs)


class stripped_op_list_for_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(stripped_op_list_for_graph, self).__init__(tf.contrib.util.stripped_op_list_for_graph, *args, **kwargs)


class copy_op_to_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(copy_op_to_graph, self).__init__(tf.contrib.copy_graph.copy_op_to_graph, *args, **kwargs)


class copy_variable_to_graph(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(copy_variable_to_graph, self).__init__(tf.contrib.copy_graph.copy_variable_to_graph, *args, **kwargs)


class get_copied_op(DelayedOperation):
    def __init__(self, *args, **kwargs):
        super(get_copied_op, self).__init__(tf.contrib.copy_graph.get_copied_op, *args, **kwargs)