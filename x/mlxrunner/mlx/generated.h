// This code is auto-generated; DO NOT EDIT.

#ifndef MLX_GENERATED_H
#define MLX_GENERATED_H

#include "dynamic.h"
#include "mlx/c/mlx.h"

#undef mlx_dtype_size
#undef mlx_array_tostring
#undef mlx_array_new
#undef mlx_array_free
#undef mlx_array_new_bool
#undef mlx_array_new_int
#undef mlx_array_new_float32
#undef mlx_array_new_float
#undef mlx_array_new_float64
#undef mlx_array_new_double
#undef mlx_array_new_complex
#undef mlx_array_new_data
#undef mlx_array_set
#undef mlx_array_set_bool
#undef mlx_array_set_int
#undef mlx_array_set_float32
#undef mlx_array_set_float
#undef mlx_array_set_float64
#undef mlx_array_set_double
#undef mlx_array_set_complex
#undef mlx_array_set_data
#undef mlx_array_itemsize
#undef mlx_array_size
#undef mlx_array_nbytes
#undef mlx_array_ndim
#undef mlx_array_shape
#undef mlx_array_strides
#undef mlx_array_dim
#undef mlx_array_dtype
#undef mlx_array_eval
#undef mlx_array_item_bool
#undef mlx_array_item_uint8
#undef mlx_array_item_uint16
#undef mlx_array_item_uint32
#undef mlx_array_item_uint64
#undef mlx_array_item_int8
#undef mlx_array_item_int16
#undef mlx_array_item_int32
#undef mlx_array_item_int64
#undef mlx_array_item_float32
#undef mlx_array_item_float64
#undef mlx_array_item_complex64
#undef mlx_array_item_float16
#undef mlx_array_item_bfloat16
#undef mlx_array_data_bool
#undef mlx_array_data_uint8
#undef mlx_array_data_uint16
#undef mlx_array_data_uint32
#undef mlx_array_data_uint64
#undef mlx_array_data_int8
#undef mlx_array_data_int16
#undef mlx_array_data_int32
#undef mlx_array_data_int64
#undef mlx_array_data_float32
#undef mlx_array_data_float64
#undef mlx_array_data_complex64
#undef mlx_array_data_float16
#undef mlx_array_data_bfloat16
#undef _mlx_array_is_available
#undef _mlx_array_wait
#undef _mlx_array_is_contiguous
#undef _mlx_array_is_row_contiguous
#undef _mlx_array_is_col_contiguous
#undef mlx_closure_new
#undef mlx_closure_free
#undef mlx_closure_new_func
#undef mlx_closure_new_func_payload
#undef mlx_closure_set
#undef mlx_closure_apply
#undef mlx_closure_new_unary
#undef mlx_closure_kwargs_new
#undef mlx_closure_kwargs_free
#undef mlx_closure_kwargs_new_func
#undef mlx_closure_kwargs_new_func_payload
#undef mlx_closure_kwargs_set
#undef mlx_closure_kwargs_apply
#undef mlx_closure_value_and_grad_new
#undef mlx_closure_value_and_grad_free
#undef mlx_closure_value_and_grad_new_func
#undef mlx_closure_value_and_grad_new_func_payload
#undef mlx_closure_value_and_grad_set
#undef mlx_closure_value_and_grad_apply
#undef mlx_closure_custom_new
#undef mlx_closure_custom_free
#undef mlx_closure_custom_new_func
#undef mlx_closure_custom_new_func_payload
#undef mlx_closure_custom_set
#undef mlx_closure_custom_apply
#undef mlx_closure_custom_jvp_new
#undef mlx_closure_custom_jvp_free
#undef mlx_closure_custom_jvp_new_func
#undef mlx_closure_custom_jvp_new_func_payload
#undef mlx_closure_custom_jvp_set
#undef mlx_closure_custom_jvp_apply
#undef mlx_closure_custom_vmap_new
#undef mlx_closure_custom_vmap_free
#undef mlx_closure_custom_vmap_new_func
#undef mlx_closure_custom_vmap_new_func_payload
#undef mlx_closure_custom_vmap_set
#undef mlx_closure_custom_vmap_apply
#undef mlx_compile
#undef mlx_detail_compile
#undef mlx_detail_compile_clear_cache
#undef mlx_detail_compile_erase
#undef mlx_disable_compile
#undef mlx_enable_compile
#undef mlx_set_compile_mode
#undef mlx_device_new
#undef mlx_device_new_type
#undef mlx_device_free
#undef mlx_device_set
#undef mlx_device_tostring
#undef mlx_device_equal
#undef mlx_device_get_index
#undef mlx_device_get_type
#undef mlx_get_default_device
#undef mlx_set_default_device
#undef mlx_distributed_group_rank
#undef mlx_distributed_group_size
#undef mlx_distributed_group_split
#undef mlx_distributed_is_available
#undef mlx_distributed_init
#undef mlx_distributed_all_gather
#undef mlx_distributed_all_max
#undef mlx_distributed_all_min
#undef mlx_distributed_all_sum
#undef mlx_distributed_recv
#undef mlx_distributed_recv_like
#undef mlx_distributed_send
#undef mlx_distributed_sum_scatter
#undef mlx_set_error_handler
#undef _mlx_error
#undef mlx_export_function
#undef mlx_export_function_kwargs
#undef mlx_function_exporter_new
#undef mlx_function_exporter_free
#undef mlx_function_exporter_apply
#undef mlx_function_exporter_apply_kwargs
#undef mlx_imported_function_new
#undef mlx_imported_function_free
#undef mlx_imported_function_apply
#undef mlx_imported_function_apply_kwargs
#undef mlx_fast_cuda_kernel_config_new
#undef mlx_fast_cuda_kernel_config_free
#undef mlx_fast_cuda_kernel_config_add_output_arg
#undef mlx_fast_cuda_kernel_config_set_grid
#undef mlx_fast_cuda_kernel_config_set_thread_group
#undef mlx_fast_cuda_kernel_config_set_init_value
#undef mlx_fast_cuda_kernel_config_set_verbose
#undef mlx_fast_cuda_kernel_config_add_template_arg_dtype
#undef mlx_fast_cuda_kernel_config_add_template_arg_int
#undef mlx_fast_cuda_kernel_config_add_template_arg_bool
#undef mlx_fast_cuda_kernel_new
#undef mlx_fast_cuda_kernel_free
#undef mlx_fast_cuda_kernel_apply
#undef mlx_fast_layer_norm
#undef mlx_fast_metal_kernel_config_new
#undef mlx_fast_metal_kernel_config_free
#undef mlx_fast_metal_kernel_config_add_output_arg
#undef mlx_fast_metal_kernel_config_set_grid
#undef mlx_fast_metal_kernel_config_set_thread_group
#undef mlx_fast_metal_kernel_config_set_init_value
#undef mlx_fast_metal_kernel_config_set_verbose
#undef mlx_fast_metal_kernel_config_add_template_arg_dtype
#undef mlx_fast_metal_kernel_config_add_template_arg_int
#undef mlx_fast_metal_kernel_config_add_template_arg_bool
#undef mlx_fast_metal_kernel_new
#undef mlx_fast_metal_kernel_free
#undef mlx_fast_metal_kernel_apply
#undef mlx_fast_rms_norm
#undef mlx_fast_rope
#undef mlx_fast_scaled_dot_product_attention
#undef mlx_fft_fft
#undef mlx_fft_fft2
#undef mlx_fft_fftn
#undef mlx_fft_fftshift
#undef mlx_fft_ifft
#undef mlx_fft_ifft2
#undef mlx_fft_ifftn
#undef mlx_fft_ifftshift
#undef mlx_fft_irfft
#undef mlx_fft_irfft2
#undef mlx_fft_irfftn
#undef mlx_fft_rfft
#undef mlx_fft_rfft2
#undef mlx_fft_rfftn
#undef mlx_io_reader_new
#undef mlx_io_reader_descriptor
#undef mlx_io_reader_tostring
#undef mlx_io_reader_free
#undef mlx_io_writer_new
#undef mlx_io_writer_descriptor
#undef mlx_io_writer_tostring
#undef mlx_io_writer_free
#undef mlx_load_reader
#undef mlx_load
#undef mlx_load_safetensors_reader
#undef mlx_load_safetensors
#undef mlx_save_writer
#undef mlx_save
#undef mlx_save_safetensors_writer
#undef mlx_save_safetensors
#undef mlx_linalg_cholesky
#undef mlx_linalg_cholesky_inv
#undef mlx_linalg_cross
#undef mlx_linalg_eig
#undef mlx_linalg_eigh
#undef mlx_linalg_eigvals
#undef mlx_linalg_eigvalsh
#undef mlx_linalg_inv
#undef mlx_linalg_lu
#undef mlx_linalg_lu_factor
#undef mlx_linalg_norm
#undef mlx_linalg_norm_matrix
#undef mlx_linalg_norm_l2
#undef mlx_linalg_pinv
#undef mlx_linalg_qr
#undef mlx_linalg_solve
#undef mlx_linalg_solve_triangular
#undef mlx_linalg_svd
#undef mlx_linalg_tri_inv
#undef mlx_map_string_to_array_new
#undef mlx_map_string_to_array_set
#undef mlx_map_string_to_array_free
#undef mlx_map_string_to_array_insert
#undef mlx_map_string_to_array_get
#undef mlx_map_string_to_array_iterator_new
#undef mlx_map_string_to_array_iterator_free
#undef mlx_map_string_to_array_iterator_next
#undef mlx_map_string_to_string_new
#undef mlx_map_string_to_string_set
#undef mlx_map_string_to_string_free
#undef mlx_map_string_to_string_insert
#undef mlx_map_string_to_string_get
#undef mlx_map_string_to_string_iterator_new
#undef mlx_map_string_to_string_iterator_free
#undef mlx_map_string_to_string_iterator_next
#undef mlx_clear_cache
#undef mlx_get_active_memory
#undef mlx_get_cache_memory
#undef mlx_get_memory_limit
#undef mlx_get_peak_memory
#undef mlx_reset_peak_memory
#undef mlx_set_cache_limit
#undef mlx_set_memory_limit
#undef mlx_set_wired_limit
#undef mlx_metal_device_info
#undef mlx_metal_is_available
#undef mlx_metal_start_capture
#undef mlx_metal_stop_capture
#undef mlx_abs
#undef mlx_add
#undef mlx_addmm
#undef mlx_all_axes
#undef mlx_all_axis
#undef mlx_all
#undef mlx_allclose
#undef mlx_any_axes
#undef mlx_any_axis
#undef mlx_any
#undef mlx_arange
#undef mlx_arccos
#undef mlx_arccosh
#undef mlx_arcsin
#undef mlx_arcsinh
#undef mlx_arctan
#undef mlx_arctan2
#undef mlx_arctanh
#undef mlx_argmax_axis
#undef mlx_argmax
#undef mlx_argmin_axis
#undef mlx_argmin
#undef mlx_argpartition_axis
#undef mlx_argpartition
#undef mlx_argsort_axis
#undef mlx_argsort
#undef mlx_array_equal
#undef mlx_as_strided
#undef mlx_astype
#undef mlx_atleast_1d
#undef mlx_atleast_2d
#undef mlx_atleast_3d
#undef mlx_bitwise_and
#undef mlx_bitwise_invert
#undef mlx_bitwise_or
#undef mlx_bitwise_xor
#undef mlx_block_masked_mm
#undef mlx_broadcast_arrays
#undef mlx_broadcast_to
#undef mlx_ceil
#undef mlx_clip
#undef mlx_concatenate_axis
#undef mlx_concatenate
#undef mlx_conjugate
#undef mlx_contiguous
#undef mlx_conv1d
#undef mlx_conv2d
#undef mlx_conv3d
#undef mlx_conv_general
#undef mlx_conv_transpose1d
#undef mlx_conv_transpose2d
#undef mlx_conv_transpose3d
#undef mlx_copy
#undef mlx_cos
#undef mlx_cosh
#undef mlx_cummax
#undef mlx_cummin
#undef mlx_cumprod
#undef mlx_cumsum
#undef mlx_degrees
#undef mlx_depends
#undef mlx_dequantize
#undef mlx_diag
#undef mlx_diagonal
#undef mlx_divide
#undef mlx_divmod
#undef mlx_einsum
#undef mlx_equal
#undef mlx_erf
#undef mlx_erfinv
#undef mlx_exp
#undef mlx_expand_dims_axes
#undef mlx_expand_dims
#undef mlx_expm1
#undef mlx_eye
#undef mlx_flatten
#undef mlx_floor
#undef mlx_floor_divide
#undef mlx_from_fp8
#undef mlx_full
#undef mlx_full_like
#undef mlx_gather
#undef mlx_gather_mm
#undef mlx_gather_qmm
#undef mlx_greater
#undef mlx_greater_equal
#undef mlx_hadamard_transform
#undef mlx_identity
#undef mlx_imag
#undef mlx_inner
#undef mlx_isclose
#undef mlx_isfinite
#undef mlx_isinf
#undef mlx_isnan
#undef mlx_isneginf
#undef mlx_isposinf
#undef mlx_kron
#undef mlx_left_shift
#undef mlx_less
#undef mlx_less_equal
#undef mlx_linspace
#undef mlx_log
#undef mlx_log10
#undef mlx_log1p
#undef mlx_log2
#undef mlx_logaddexp
#undef mlx_logcumsumexp
#undef mlx_logical_and
#undef mlx_logical_not
#undef mlx_logical_or
#undef mlx_logsumexp_axes
#undef mlx_logsumexp_axis
#undef mlx_logsumexp
#undef mlx_masked_scatter
#undef mlx_matmul
#undef mlx_max_axes
#undef mlx_max_axis
#undef mlx_max
#undef mlx_maximum
#undef mlx_mean_axes
#undef mlx_mean_axis
#undef mlx_mean
#undef mlx_median
#undef mlx_meshgrid
#undef mlx_min_axes
#undef mlx_min_axis
#undef mlx_min
#undef mlx_minimum
#undef mlx_moveaxis
#undef mlx_multiply
#undef mlx_nan_to_num
#undef mlx_negative
#undef mlx_not_equal
#undef mlx_number_of_elements
#undef mlx_ones
#undef mlx_ones_like
#undef mlx_outer
#undef mlx_pad
#undef mlx_pad_symmetric
#undef mlx_partition_axis
#undef mlx_partition
#undef mlx_power
#undef mlx_prod_axes
#undef mlx_prod_axis
#undef mlx_prod
#undef mlx_put_along_axis
#undef mlx_quantize
#undef mlx_quantized_matmul
#undef mlx_radians
#undef mlx_real
#undef mlx_reciprocal
#undef mlx_remainder
#undef mlx_repeat_axis
#undef mlx_repeat
#undef mlx_reshape
#undef mlx_right_shift
#undef mlx_roll_axis
#undef mlx_roll_axes
#undef mlx_roll
#undef mlx_round
#undef mlx_rsqrt
#undef mlx_scatter
#undef mlx_scatter_add
#undef mlx_scatter_add_axis
#undef mlx_scatter_max
#undef mlx_scatter_min
#undef mlx_scatter_prod
#undef mlx_segmented_mm
#undef mlx_sigmoid
#undef mlx_sign
#undef mlx_sin
#undef mlx_sinh
#undef mlx_slice
#undef mlx_slice_dynamic
#undef mlx_slice_update
#undef mlx_slice_update_dynamic
#undef mlx_softmax_axes
#undef mlx_softmax_axis
#undef mlx_softmax
#undef mlx_sort_axis
#undef mlx_sort
#undef mlx_split
#undef mlx_split_sections
#undef mlx_sqrt
#undef mlx_square
#undef mlx_squeeze_axes
#undef mlx_squeeze_axis
#undef mlx_squeeze
#undef mlx_stack_axis
#undef mlx_stack
#undef mlx_std_axes
#undef mlx_std_axis
#undef mlx_std
#undef mlx_stop_gradient
#undef mlx_subtract
#undef mlx_sum_axes
#undef mlx_sum_axis
#undef mlx_sum
#undef mlx_swapaxes
#undef mlx_take_axis
#undef mlx_take
#undef mlx_take_along_axis
#undef mlx_tan
#undef mlx_tanh
#undef mlx_tensordot
#undef mlx_tensordot_axis
#undef mlx_tile
#undef mlx_to_fp8
#undef mlx_topk_axis
#undef mlx_topk
#undef mlx_trace
#undef mlx_transpose_axes
#undef mlx_transpose
#undef mlx_tri
#undef mlx_tril
#undef mlx_triu
#undef mlx_unflatten
#undef mlx_var_axes
#undef mlx_var_axis
#undef mlx_var
#undef mlx_view
#undef mlx_where
#undef mlx_zeros
#undef mlx_zeros_like
#undef mlx_random_bernoulli
#undef mlx_random_bits
#undef mlx_random_categorical_shape
#undef mlx_random_categorical_num_samples
#undef mlx_random_categorical
#undef mlx_random_gumbel
#undef mlx_random_key
#undef mlx_random_laplace
#undef mlx_random_multivariate_normal
#undef mlx_random_normal_broadcast
#undef mlx_random_normal
#undef mlx_random_permutation
#undef mlx_random_permutation_arange
#undef mlx_random_randint
#undef mlx_random_seed
#undef mlx_random_split_num
#undef mlx_random_split
#undef mlx_random_truncated_normal
#undef mlx_random_uniform
#undef mlx_stream_new
#undef mlx_stream_new_device
#undef mlx_stream_set
#undef mlx_stream_free
#undef mlx_stream_tostring
#undef mlx_stream_equal
#undef mlx_stream_get_device
#undef mlx_stream_get_index
#undef mlx_synchronize
#undef mlx_get_default_stream
#undef mlx_set_default_stream
#undef mlx_default_cpu_stream_new
#undef mlx_default_gpu_stream_new
#undef mlx_string_new
#undef mlx_string_new_data
#undef mlx_string_set
#undef mlx_string_data
#undef mlx_string_free
#undef mlx_detail_vmap_replace
#undef mlx_detail_vmap_trace
#undef mlx_async_eval
#undef mlx_checkpoint
#undef mlx_custom_function
#undef mlx_custom_vjp
#undef mlx_eval
#undef mlx_jvp
#undef mlx_value_and_grad
#undef mlx_vjp
#undef mlx_vector_array_new
#undef mlx_vector_array_set
#undef mlx_vector_array_free
#undef mlx_vector_array_new_data
#undef mlx_vector_array_new_value
#undef mlx_vector_array_set_data
#undef mlx_vector_array_set_value
#undef mlx_vector_array_append_data
#undef mlx_vector_array_append_value
#undef mlx_vector_array_size
#undef mlx_vector_array_get
#undef mlx_vector_vector_array_new
#undef mlx_vector_vector_array_set
#undef mlx_vector_vector_array_free
#undef mlx_vector_vector_array_new_data
#undef mlx_vector_vector_array_new_value
#undef mlx_vector_vector_array_set_data
#undef mlx_vector_vector_array_set_value
#undef mlx_vector_vector_array_append_data
#undef mlx_vector_vector_array_append_value
#undef mlx_vector_vector_array_size
#undef mlx_vector_vector_array_get
#undef mlx_vector_int_new
#undef mlx_vector_int_set
#undef mlx_vector_int_free
#undef mlx_vector_int_new_data
#undef mlx_vector_int_new_value
#undef mlx_vector_int_set_data
#undef mlx_vector_int_set_value
#undef mlx_vector_int_append_data
#undef mlx_vector_int_append_value
#undef mlx_vector_int_size
#undef mlx_vector_int_get
#undef mlx_vector_string_new
#undef mlx_vector_string_set
#undef mlx_vector_string_free
#undef mlx_vector_string_new_data
#undef mlx_vector_string_new_value
#undef mlx_vector_string_set_data
#undef mlx_vector_string_set_value
#undef mlx_vector_string_append_data
#undef mlx_vector_string_append_value
#undef mlx_vector_string_size
#undef mlx_vector_string_get
#undef mlx_version

extern size_t (*mlx_dtype_size_)(mlx_dtype dtype);
extern int (*mlx_array_tostring_)(mlx_string* str, const mlx_array arr);
extern mlx_array (*mlx_array_new_)(void);
extern int (*mlx_array_free_)(mlx_array arr);
extern mlx_array (*mlx_array_new_bool_)(bool val);
extern mlx_array (*mlx_array_new_int_)(int val);
extern mlx_array (*mlx_array_new_float32_)(float val);
extern mlx_array (*mlx_array_new_float_)(float val);
extern mlx_array (*mlx_array_new_float64_)(double val);
extern mlx_array (*mlx_array_new_double_)(double val);
extern mlx_array (*mlx_array_new_complex_)(float real_val, float imag_val);
extern mlx_array (*mlx_array_new_data_)(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
extern int (*mlx_array_set_)(mlx_array* arr, const mlx_array src);
extern int (*mlx_array_set_bool_)(mlx_array* arr, bool val);
extern int (*mlx_array_set_int_)(mlx_array* arr, int val);
extern int (*mlx_array_set_float32_)(mlx_array* arr, float val);
extern int (*mlx_array_set_float_)(mlx_array* arr, float val);
extern int (*mlx_array_set_float64_)(mlx_array* arr, double val);
extern int (*mlx_array_set_double_)(mlx_array* arr, double val);
extern int (*mlx_array_set_complex_)(mlx_array* arr, float real_val, float imag_val);
extern int (*mlx_array_set_data_)(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
extern size_t (*mlx_array_itemsize_)(const mlx_array arr);
extern size_t (*mlx_array_size_)(const mlx_array arr);
extern size_t (*mlx_array_nbytes_)(const mlx_array arr);
extern size_t (*mlx_array_ndim_)(const mlx_array arr);
extern const int * (*mlx_array_shape_)(const mlx_array arr);
extern const size_t * (*mlx_array_strides_)(const mlx_array arr);
extern int (*mlx_array_dim_)(const mlx_array arr, int dim);
extern mlx_dtype (*mlx_array_dtype_)(const mlx_array arr);
extern int (*mlx_array_eval_)(mlx_array arr);
extern int (*mlx_array_item_bool_)(bool* res, const mlx_array arr);
extern int (*mlx_array_item_uint8_)(uint8_t* res, const mlx_array arr);
extern int (*mlx_array_item_uint16_)(uint16_t* res, const mlx_array arr);
extern int (*mlx_array_item_uint32_)(uint32_t* res, const mlx_array arr);
extern int (*mlx_array_item_uint64_)(uint64_t* res, const mlx_array arr);
extern int (*mlx_array_item_int8_)(int8_t* res, const mlx_array arr);
extern int (*mlx_array_item_int16_)(int16_t* res, const mlx_array arr);
extern int (*mlx_array_item_int32_)(int32_t* res, const mlx_array arr);
extern int (*mlx_array_item_int64_)(int64_t* res, const mlx_array arr);
extern int (*mlx_array_item_float32_)(float* res, const mlx_array arr);
extern int (*mlx_array_item_float64_)(double* res, const mlx_array arr);
extern int (*mlx_array_item_complex64_)(float _Complex* res, const mlx_array arr);
extern int (*mlx_array_item_float16_)(float16_t* res, const mlx_array arr);
extern int (*mlx_array_item_bfloat16_)(bfloat16_t* res, const mlx_array arr);
extern const bool * (*mlx_array_data_bool_)(const mlx_array arr);
extern const uint8_t * (*mlx_array_data_uint8_)(const mlx_array arr);
extern const uint16_t * (*mlx_array_data_uint16_)(const mlx_array arr);
extern const uint32_t * (*mlx_array_data_uint32_)(const mlx_array arr);
extern const uint64_t * (*mlx_array_data_uint64_)(const mlx_array arr);
extern const int8_t * (*mlx_array_data_int8_)(const mlx_array arr);
extern const int16_t * (*mlx_array_data_int16_)(const mlx_array arr);
extern const int32_t * (*mlx_array_data_int32_)(const mlx_array arr);
extern const int64_t * (*mlx_array_data_int64_)(const mlx_array arr);
extern const float * (*mlx_array_data_float32_)(const mlx_array arr);
extern const double * (*mlx_array_data_float64_)(const mlx_array arr);
extern const float _Complex * (*mlx_array_data_complex64_)(const mlx_array arr);
extern const float16_t * (*mlx_array_data_float16_)(const mlx_array arr);
extern const bfloat16_t * (*mlx_array_data_bfloat16_)(const mlx_array arr);
extern int (*_mlx_array_is_available_)(bool* res, const mlx_array arr);
extern int (*_mlx_array_wait_)(const mlx_array arr);
extern int (*_mlx_array_is_contiguous_)(bool* res, const mlx_array arr);
extern int (*_mlx_array_is_row_contiguous_)(bool* res, const mlx_array arr);
extern int (*_mlx_array_is_col_contiguous_)(bool* res, const mlx_array arr);
extern mlx_closure (*mlx_closure_new_)(void);
extern int (*mlx_closure_free_)(mlx_closure cls);
extern mlx_closure (*mlx_closure_new_func_)(
    int (*fun)(mlx_vector_array*, const mlx_vector_array));
extern mlx_closure (*mlx_closure_new_func_payload_)(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*));
extern int (*mlx_closure_set_)(mlx_closure* cls, const mlx_closure src);
extern int (*mlx_closure_apply_)(
    mlx_vector_array* res,
    mlx_closure cls,
    const mlx_vector_array input);
extern mlx_closure (*mlx_closure_new_unary_)(int (*fun)(mlx_array*, const mlx_array));
extern mlx_closure_kwargs (*mlx_closure_kwargs_new_)(void);
extern int (*mlx_closure_kwargs_free_)(mlx_closure_kwargs cls);
extern mlx_closure_kwargs (*mlx_closure_kwargs_new_func_)(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_map_string_to_array));
extern mlx_closure_kwargs (*mlx_closure_kwargs_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array,
        void*),
    void* payload,
    void (*dtor)(void*));
extern int (*mlx_closure_kwargs_set_)(
    mlx_closure_kwargs* cls,
    const mlx_closure_kwargs src);
extern int (*mlx_closure_kwargs_apply_)(
    mlx_vector_array* res,
    mlx_closure_kwargs cls,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1);
extern mlx_closure_value_and_grad (*mlx_closure_value_and_grad_new_)(void);
extern int (*mlx_closure_value_and_grad_free_)(mlx_closure_value_and_grad cls);
extern mlx_closure_value_and_grad (*mlx_closure_value_and_grad_new_func_)(
    int (*fun)(mlx_vector_array*, mlx_vector_array*, const mlx_vector_array));
extern mlx_closure_value_and_grad (*mlx_closure_value_and_grad_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_array*,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*));
extern int (*mlx_closure_value_and_grad_set_)(
    mlx_closure_value_and_grad* cls,
    const mlx_closure_value_and_grad src);
extern int (*mlx_closure_value_and_grad_apply_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    mlx_closure_value_and_grad cls,
    const mlx_vector_array input);
extern mlx_closure_custom (*mlx_closure_custom_new_)(void);
extern int (*mlx_closure_custom_free_)(mlx_closure_custom cls);
extern mlx_closure_custom (*mlx_closure_custom_new_func_)(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const mlx_vector_array));
extern mlx_closure_custom (*mlx_closure_custom_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*));
extern int (*mlx_closure_custom_set_)(
    mlx_closure_custom* cls,
    const mlx_closure_custom src);
extern int (*mlx_closure_custom_apply_)(
    mlx_vector_array* res,
    mlx_closure_custom cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2);
extern mlx_closure_custom_jvp (*mlx_closure_custom_jvp_new_)(void);
extern int (*mlx_closure_custom_jvp_free_)(mlx_closure_custom_jvp cls);
extern mlx_closure_custom_jvp (*mlx_closure_custom_jvp_new_func_)(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const int*,
    size_t _num));
extern mlx_closure_custom_jvp (*mlx_closure_custom_jvp_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*));
extern int (*mlx_closure_custom_jvp_set_)(
    mlx_closure_custom_jvp* cls,
    const mlx_closure_custom_jvp src);
extern int (*mlx_closure_custom_jvp_apply_)(
    mlx_vector_array* res,
    mlx_closure_custom_jvp cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const int* input_2,
    size_t input_2_num);
extern mlx_closure_custom_vmap (*mlx_closure_custom_vmap_new_)(void);
extern int (*mlx_closure_custom_vmap_free_)(mlx_closure_custom_vmap cls);
extern mlx_closure_custom_vmap (*mlx_closure_custom_vmap_new_func_)(int (*fun)(
    mlx_vector_array*,
    mlx_vector_int*,
    const mlx_vector_array,
    const int*,
    size_t _num));
extern mlx_closure_custom_vmap (*mlx_closure_custom_vmap_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*));
extern int (*mlx_closure_custom_vmap_set_)(
    mlx_closure_custom_vmap* cls,
    const mlx_closure_custom_vmap src);
extern int (*mlx_closure_custom_vmap_apply_)(
    mlx_vector_array* res_0,
    mlx_vector_int* res_1,
    mlx_closure_custom_vmap cls,
    const mlx_vector_array input_0,
    const int* input_1,
    size_t input_1_num);
extern int (*mlx_compile_)(mlx_closure* res, const mlx_closure fun, bool shapeless);
extern int (*mlx_detail_compile_)(
    mlx_closure* res,
    const mlx_closure fun,
    uintptr_t fun_id,
    bool shapeless,
    const uint64_t* constants,
    size_t constants_num);
extern int (*mlx_detail_compile_clear_cache_)(void);
extern int (*mlx_detail_compile_erase_)(uintptr_t fun_id);
extern int (*mlx_disable_compile_)(void);
extern int (*mlx_enable_compile_)(void);
extern int (*mlx_set_compile_mode_)(mlx_compile_mode mode);
extern mlx_device (*mlx_device_new_)(void);
extern mlx_device (*mlx_device_new_type_)(mlx_device_type type, int index);
extern int (*mlx_device_free_)(mlx_device dev);
extern int (*mlx_device_set_)(mlx_device* dev, const mlx_device src);
extern int (*mlx_device_tostring_)(mlx_string* str, mlx_device dev);
extern bool (*mlx_device_equal_)(mlx_device lhs, mlx_device rhs);
extern int (*mlx_device_get_index_)(int* index, mlx_device dev);
extern int (*mlx_device_get_type_)(mlx_device_type* type, mlx_device dev);
extern int (*mlx_get_default_device_)(mlx_device* dev);
extern int (*mlx_set_default_device_)(mlx_device dev);
extern int (*mlx_distributed_group_rank_)(mlx_distributed_group group);
extern int (*mlx_distributed_group_size_)(mlx_distributed_group group);
extern mlx_distributed_group (*mlx_distributed_group_split_)(mlx_distributed_group group, int color, int key);
extern bool (*mlx_distributed_is_available_)(void);
extern mlx_distributed_group (*mlx_distributed_init_)(bool strict);
extern int (*mlx_distributed_all_gather_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream S);
extern int (*mlx_distributed_all_max_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern int (*mlx_distributed_all_min_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern int (*mlx_distributed_all_sum_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern int (*mlx_distributed_recv_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern int (*mlx_distributed_recv_like_)(
    mlx_array* res,
    const mlx_array x,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern int (*mlx_distributed_send_)(
    mlx_array* res,
    const mlx_array x,
    int dst,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern int (*mlx_distributed_sum_scatter_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
extern void (*mlx_set_error_handler_)(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*));
extern void (*_mlx_error_)(const char* file, const int line, const char* fmt, ...);
extern int (*mlx_export_function_)(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless);
extern int (*mlx_export_function_kwargs_)(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless);
extern mlx_function_exporter (*mlx_function_exporter_new_)(
    const char* file,
    const mlx_closure fun,
    bool shapeless);
extern int (*mlx_function_exporter_free_)(mlx_function_exporter xfunc);
extern int (*mlx_function_exporter_apply_)(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args);
extern int (*mlx_function_exporter_apply_kwargs_)(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);
extern mlx_imported_function (*mlx_imported_function_new_)(const char* file);
extern int (*mlx_imported_function_free_)(mlx_imported_function xfunc);
extern int (*mlx_imported_function_apply_)(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args);
extern int (*mlx_imported_function_apply_kwargs_)(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);
extern mlx_fast_cuda_kernel_config (*mlx_fast_cuda_kernel_config_new_)(void);
extern void (*mlx_fast_cuda_kernel_config_free_)(mlx_fast_cuda_kernel_config cls);
extern int (*mlx_fast_cuda_kernel_config_add_output_arg_)(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
extern int (*mlx_fast_cuda_kernel_config_set_grid_)(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
extern int (*mlx_fast_cuda_kernel_config_set_thread_group_)(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
extern int (*mlx_fast_cuda_kernel_config_set_init_value_)(
    mlx_fast_cuda_kernel_config cls,
    float value);
extern int (*mlx_fast_cuda_kernel_config_set_verbose_)(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
extern int (*mlx_fast_cuda_kernel_config_add_template_arg_dtype_)(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
extern int (*mlx_fast_cuda_kernel_config_add_template_arg_int_)(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
extern int (*mlx_fast_cuda_kernel_config_add_template_arg_bool_)(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);
extern mlx_fast_cuda_kernel (*mlx_fast_cuda_kernel_new_)(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);
extern void (*mlx_fast_cuda_kernel_free_)(mlx_fast_cuda_kernel cls);
extern int (*mlx_fast_cuda_kernel_apply_)(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);
extern int (*mlx_fast_layer_norm_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);
extern mlx_fast_metal_kernel_config (*mlx_fast_metal_kernel_config_new_)(void);
extern void (*mlx_fast_metal_kernel_config_free_)(mlx_fast_metal_kernel_config cls);
extern int (*mlx_fast_metal_kernel_config_add_output_arg_)(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
extern int (*mlx_fast_metal_kernel_config_set_grid_)(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
extern int (*mlx_fast_metal_kernel_config_set_thread_group_)(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
extern int (*mlx_fast_metal_kernel_config_set_init_value_)(
    mlx_fast_metal_kernel_config cls,
    float value);
extern int (*mlx_fast_metal_kernel_config_set_verbose_)(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
extern int (*mlx_fast_metal_kernel_config_add_template_arg_dtype_)(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
extern int (*mlx_fast_metal_kernel_config_add_template_arg_int_)(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
extern int (*mlx_fast_metal_kernel_config_add_template_arg_bool_)(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);
extern mlx_fast_metal_kernel (*mlx_fast_metal_kernel_new_)(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);
extern void (*mlx_fast_metal_kernel_free_)(mlx_fast_metal_kernel cls);
extern int (*mlx_fast_metal_kernel_apply_)(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);
extern int (*mlx_fast_rms_norm_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
extern int (*mlx_fast_rope_)(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
extern int (*mlx_fast_scaled_dot_product_attention_)(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s);
extern int (*mlx_fft_fft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
extern int (*mlx_fft_fft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_fftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_fftshift_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_ifft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
extern int (*mlx_fft_ifft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_ifftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_ifftshift_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_irfft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
extern int (*mlx_fft_irfft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_irfftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_rfft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
extern int (*mlx_fft_rfft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_fft_rfftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern mlx_io_reader (*mlx_io_reader_new_)(void* desc, mlx_io_vtable vtable);
extern int (*mlx_io_reader_descriptor_)(void** desc_, mlx_io_reader io);
extern int (*mlx_io_reader_tostring_)(mlx_string* str_, mlx_io_reader io);
extern int (*mlx_io_reader_free_)(mlx_io_reader io);
extern mlx_io_writer (*mlx_io_writer_new_)(void* desc, mlx_io_vtable vtable);
extern int (*mlx_io_writer_descriptor_)(void** desc_, mlx_io_writer io);
extern int (*mlx_io_writer_tostring_)(mlx_string* str_, mlx_io_writer io);
extern int (*mlx_io_writer_free_)(mlx_io_writer io);
extern int (*mlx_load_reader_)(
    mlx_array* res,
    mlx_io_reader in_stream,
    const mlx_stream s);
extern int (*mlx_load_)(mlx_array* res, const char* file, const mlx_stream s);
extern int (*mlx_load_safetensors_reader_)(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    mlx_io_reader in_stream,
    const mlx_stream s);
extern int (*mlx_load_safetensors_)(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    const char* file,
    const mlx_stream s);
extern int (*mlx_save_writer_)(mlx_io_writer out_stream, const mlx_array a);
extern int (*mlx_save_)(const char* file, const mlx_array a);
extern int (*mlx_save_safetensors_writer_)(
    mlx_io_writer in_stream,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata);
extern int (*mlx_save_safetensors_)(
    const char* file,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata);
extern int (*mlx_linalg_cholesky_)(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
extern int (*mlx_linalg_cholesky_inv_)(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
extern int (*mlx_linalg_cross_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s);
extern int (*mlx_linalg_eig_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
extern int (*mlx_linalg_eigh_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s);
extern int (*mlx_linalg_eigvals_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_linalg_eigvalsh_)(
    mlx_array* res,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s);
extern int (*mlx_linalg_inv_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_linalg_lu_)(mlx_vector_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_linalg_lu_factor_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
extern int (*mlx_linalg_norm_)(
    mlx_array* res,
    const mlx_array a,
    double ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_linalg_norm_matrix_)(
    mlx_array* res,
    const mlx_array a,
    const char* ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_linalg_norm_l2_)(
    mlx_array* res,
    const mlx_array a,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_linalg_pinv_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_linalg_qr_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
extern int (*mlx_linalg_solve_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_linalg_solve_triangular_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool upper,
    const mlx_stream s);
extern int (*mlx_linalg_svd_)(
    mlx_vector_array* res,
    const mlx_array a,
    bool compute_uv,
    const mlx_stream s);
extern int (*mlx_linalg_tri_inv_)(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
extern mlx_map_string_to_array (*mlx_map_string_to_array_new_)(void);
extern int (*mlx_map_string_to_array_set_)(
    mlx_map_string_to_array* map,
    const mlx_map_string_to_array src);
extern int (*mlx_map_string_to_array_free_)(mlx_map_string_to_array map);
extern int (*mlx_map_string_to_array_insert_)(
    mlx_map_string_to_array map,
    const char* key,
    const mlx_array value);
extern int (*mlx_map_string_to_array_get_)(
    mlx_array* value,
    const mlx_map_string_to_array map,
    const char* key);
extern mlx_map_string_to_array_iterator (*mlx_map_string_to_array_iterator_new_)(
    mlx_map_string_to_array map);
extern int (*mlx_map_string_to_array_iterator_free_)(mlx_map_string_to_array_iterator it);
extern int (*mlx_map_string_to_array_iterator_next_)(
    const char** key,
    mlx_array* value,
    mlx_map_string_to_array_iterator it);
extern mlx_map_string_to_string (*mlx_map_string_to_string_new_)(void);
extern int (*mlx_map_string_to_string_set_)(
    mlx_map_string_to_string* map,
    const mlx_map_string_to_string src);
extern int (*mlx_map_string_to_string_free_)(mlx_map_string_to_string map);
extern int (*mlx_map_string_to_string_insert_)(
    mlx_map_string_to_string map,
    const char* key,
    const char* value);
extern int (*mlx_map_string_to_string_get_)(
    const char** value,
    const mlx_map_string_to_string map,
    const char* key);
extern mlx_map_string_to_string_iterator (*mlx_map_string_to_string_iterator_new_)(
    mlx_map_string_to_string map);
extern int (*mlx_map_string_to_string_iterator_free_)(
    mlx_map_string_to_string_iterator it);
extern int (*mlx_map_string_to_string_iterator_next_)(
    const char** key,
    const char** value,
    mlx_map_string_to_string_iterator it);
extern int (*mlx_clear_cache_)(void);
extern int (*mlx_get_active_memory_)(size_t* res);
extern int (*mlx_get_cache_memory_)(size_t* res);
extern int (*mlx_get_memory_limit_)(size_t* res);
extern int (*mlx_get_peak_memory_)(size_t* res);
extern int (*mlx_reset_peak_memory_)(void);
extern int (*mlx_set_cache_limit_)(size_t* res, size_t limit);
extern int (*mlx_set_memory_limit_)(size_t* res, size_t limit);
extern int (*mlx_set_wired_limit_)(size_t* res, size_t limit);
extern mlx_metal_device_info_t (*mlx_metal_device_info_)(void);
extern int (*mlx_metal_is_available_)(bool* res);
extern int (*mlx_metal_start_capture_)(const char* path);
extern int (*mlx_metal_stop_capture_)(void);
extern int (*mlx_abs_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_add_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_addmm_)(
    mlx_array* res,
    const mlx_array c,
    const mlx_array a,
    const mlx_array b,
    float alpha,
    float beta,
    const mlx_stream s);
extern int (*mlx_all_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_all_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_all_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_allclose_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s);
extern int (*mlx_any_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_any_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_any_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_arange_)(
    mlx_array* res,
    double start,
    double stop,
    double step,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_arccos_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_arccosh_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_arcsin_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_arcsinh_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_arctan_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_arctan2_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_arctanh_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_argmax_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_argmax_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_argmin_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_argmin_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_argpartition_axis_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s);
extern int (*mlx_argpartition_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s);
extern int (*mlx_argsort_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
extern int (*mlx_argsort_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_array_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool equal_nan,
    const mlx_stream s);
extern int (*mlx_as_strided_)(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const int64_t* strides,
    size_t strides_num,
    size_t offset,
    const mlx_stream s);
extern int (*mlx_astype_)(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_atleast_1d_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_atleast_2d_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_atleast_3d_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_bitwise_and_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_bitwise_invert_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_bitwise_or_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_bitwise_xor_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_block_masked_mm_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int block_size,
    const mlx_array mask_out /* may be null */,
    const mlx_array mask_lhs /* may be null */,
    const mlx_array mask_rhs /* may be null */,
    const mlx_stream s);
extern int (*mlx_broadcast_arrays_)(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_stream s);
extern int (*mlx_broadcast_to_)(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s);
extern int (*mlx_ceil_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_clip_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array a_min /* may be null */,
    const mlx_array a_max /* may be null */,
    const mlx_stream s);
extern int (*mlx_concatenate_axis_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s);
extern int (*mlx_concatenate_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s);
extern int (*mlx_conjugate_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_contiguous_)(
    mlx_array* res,
    const mlx_array a,
    bool allow_col_major,
    const mlx_stream s);
extern int (*mlx_conv1d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int groups,
    const mlx_stream s);
extern int (*mlx_conv2d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int padding_0,
    int padding_1,
    int dilation_0,
    int dilation_1,
    int groups,
    const mlx_stream s);
extern int (*mlx_conv3d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int stride_2,
    int padding_0,
    int padding_1,
    int padding_2,
    int dilation_0,
    int dilation_1,
    int dilation_2,
    int groups,
    const mlx_stream s);
extern int (*mlx_conv_general_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    const int* stride,
    size_t stride_num,
    const int* padding_lo,
    size_t padding_lo_num,
    const int* padding_hi,
    size_t padding_hi_num,
    const int* kernel_dilation,
    size_t kernel_dilation_num,
    const int* input_dilation,
    size_t input_dilation_num,
    int groups,
    bool flip,
    const mlx_stream s);
extern int (*mlx_conv_transpose1d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int output_padding,
    int groups,
    const mlx_stream s);
extern int (*mlx_conv_transpose2d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int padding_0,
    int padding_1,
    int dilation_0,
    int dilation_1,
    int output_padding_0,
    int output_padding_1,
    int groups,
    const mlx_stream s);
extern int (*mlx_conv_transpose3d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int stride_2,
    int padding_0,
    int padding_1,
    int padding_2,
    int dilation_0,
    int dilation_1,
    int dilation_2,
    int output_padding_0,
    int output_padding_1,
    int output_padding_2,
    int groups,
    const mlx_stream s);
extern int (*mlx_copy_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_cos_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_cosh_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_cummax_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
extern int (*mlx_cummin_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
extern int (*mlx_cumprod_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
extern int (*mlx_cumsum_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
extern int (*mlx_degrees_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_depends_)(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array dependencies);
extern int (*mlx_dequantize_)(
    mlx_array* res,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    mlx_optional_dtype dtype,
    const mlx_stream s);
extern int (*mlx_diag_)(mlx_array* res, const mlx_array a, int k, const mlx_stream s);
extern int (*mlx_diagonal_)(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    const mlx_stream s);
extern int (*mlx_divide_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_divmod_)(
    mlx_vector_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_einsum_)(
    mlx_array* res,
    const char* subscripts,
    const mlx_vector_array operands,
    const mlx_stream s);
extern int (*mlx_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_erf_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_erfinv_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_exp_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_expand_dims_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_expand_dims_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
extern int (*mlx_expm1_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_eye_)(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_flatten_)(
    mlx_array* res,
    const mlx_array a,
    int start_axis,
    int end_axis,
    const mlx_stream s);
extern int (*mlx_floor_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_floor_divide_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_from_fp8_)(
    mlx_array* res,
    const mlx_array x,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_full_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_full_like_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_gather_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const int* axes,
    size_t axes_num,
    const int* slice_sizes,
    size_t slice_sizes_num,
    const mlx_stream s);
extern int (*mlx_gather_mm_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool sorted_indices,
    const mlx_stream s);
extern int (*mlx_gather_qmm_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    bool sorted_indices,
    const mlx_stream s);
extern int (*mlx_greater_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_greater_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_hadamard_transform_)(
    mlx_array* res,
    const mlx_array a,
    mlx_optional_float scale,
    const mlx_stream s);
extern int (*mlx_identity_)(mlx_array* res, int n, mlx_dtype dtype, const mlx_stream s);
extern int (*mlx_imag_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_inner_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_isclose_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s);
extern int (*mlx_isfinite_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_isinf_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_isnan_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_isneginf_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_isposinf_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_kron_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_left_shift_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_less_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_less_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_linspace_)(
    mlx_array* res,
    double start,
    double stop,
    int num,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_log_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_log10_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_log1p_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_log2_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_logaddexp_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_logcumsumexp_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
extern int (*mlx_logical_and_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_logical_not_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_logical_or_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_logsumexp_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_logsumexp_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_logsumexp_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_masked_scatter_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array mask,
    const mlx_array src,
    const mlx_stream s);
extern int (*mlx_matmul_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_max_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_max_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_max_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_maximum_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_mean_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_mean_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_mean_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_median_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_meshgrid_)(
    mlx_vector_array* res,
    const mlx_vector_array arrays,
    bool sparse,
    const char* indexing,
    const mlx_stream s);
extern int (*mlx_min_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_min_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_min_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_minimum_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_moveaxis_)(
    mlx_array* res,
    const mlx_array a,
    int source,
    int destination,
    const mlx_stream s);
extern int (*mlx_multiply_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_nan_to_num_)(
    mlx_array* res,
    const mlx_array a,
    float nan,
    mlx_optional_float posinf,
    mlx_optional_float neginf,
    const mlx_stream s);
extern int (*mlx_negative_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_not_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_number_of_elements_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool inverted,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_ones_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_ones_like_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_outer_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_pad_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const int* low_pad_size,
    size_t low_pad_size_num,
    const int* high_pad_size,
    size_t high_pad_size_num,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s);
extern int (*mlx_pad_symmetric_)(
    mlx_array* res,
    const mlx_array a,
    int pad_width,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s);
extern int (*mlx_partition_axis_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s);
extern int (*mlx_partition_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s);
extern int (*mlx_power_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_prod_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_prod_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_prod_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_put_along_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s);
extern int (*mlx_quantize_)(
    mlx_vector_array* res,
    const mlx_array w,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s);
extern int (*mlx_quantized_matmul_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s);
extern int (*mlx_radians_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_real_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_reciprocal_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_remainder_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_repeat_axis_)(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    int axis,
    const mlx_stream s);
extern int (*mlx_repeat_)(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    const mlx_stream s);
extern int (*mlx_reshape_)(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s);
extern int (*mlx_right_shift_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_roll_axis_)(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    int axis,
    const mlx_stream s);
extern int (*mlx_roll_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_roll_)(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const mlx_stream s);
extern int (*mlx_round_)(
    mlx_array* res,
    const mlx_array a,
    int decimals,
    const mlx_stream s);
extern int (*mlx_rsqrt_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_scatter_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_scatter_add_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_scatter_add_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s);
extern int (*mlx_scatter_max_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_scatter_min_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_scatter_prod_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_segmented_mm_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array segments,
    const mlx_stream s);
extern int (*mlx_sigmoid_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_sign_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_sin_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_sinh_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_slice_)(
    mlx_array* res,
    const mlx_array a,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s);
extern int (*mlx_slice_dynamic_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const int* slice_size,
    size_t slice_size_num,
    const mlx_stream s);
extern int (*mlx_slice_update_)(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s);
extern int (*mlx_slice_update_dynamic_)(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_softmax_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool precise,
    const mlx_stream s);
extern int (*mlx_softmax_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool precise,
    const mlx_stream s);
extern int (*mlx_softmax_)(
    mlx_array* res,
    const mlx_array a,
    bool precise,
    const mlx_stream s);
extern int (*mlx_sort_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
extern int (*mlx_sort_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_split_)(
    mlx_vector_array* res,
    const mlx_array a,
    int num_splits,
    int axis,
    const mlx_stream s);
extern int (*mlx_split_sections_)(
    mlx_vector_array* res,
    const mlx_array a,
    const int* indices,
    size_t indices_num,
    int axis,
    const mlx_stream s);
extern int (*mlx_sqrt_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_square_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_squeeze_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_squeeze_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
extern int (*mlx_squeeze_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_stack_axis_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s);
extern int (*mlx_stack_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s);
extern int (*mlx_std_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s);
extern int (*mlx_std_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s);
extern int (*mlx_std_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s);
extern int (*mlx_stop_gradient_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_subtract_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
extern int (*mlx_sum_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_sum_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_sum_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
extern int (*mlx_swapaxes_)(
    mlx_array* res,
    const mlx_array a,
    int axis1,
    int axis2,
    const mlx_stream s);
extern int (*mlx_take_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s);
extern int (*mlx_take_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_stream s);
extern int (*mlx_take_along_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s);
extern int (*mlx_tan_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_tanh_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_tensordot_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const int* axes_a,
    size_t axes_a_num,
    const int* axes_b,
    size_t axes_b_num,
    const mlx_stream s);
extern int (*mlx_tensordot_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s);
extern int (*mlx_tile_)(
    mlx_array* res,
    const mlx_array arr,
    const int* reps,
    size_t reps_num,
    const mlx_stream s);
extern int (*mlx_to_fp8_)(mlx_array* res, const mlx_array x, const mlx_stream s);
extern int (*mlx_topk_axis_)(
    mlx_array* res,
    const mlx_array a,
    int k,
    int axis,
    const mlx_stream s);
extern int (*mlx_topk_)(mlx_array* res, const mlx_array a, int k, const mlx_stream s);
extern int (*mlx_trace_)(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_transpose_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
extern int (*mlx_transpose_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_tri_)(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype type,
    const mlx_stream s);
extern int (*mlx_tril_)(mlx_array* res, const mlx_array x, int k, const mlx_stream s);
extern int (*mlx_triu_)(mlx_array* res, const mlx_array x, int k, const mlx_stream s);
extern int (*mlx_unflatten_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_stream s);
extern int (*mlx_var_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s);
extern int (*mlx_var_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s);
extern int (*mlx_var_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s);
extern int (*mlx_view_)(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_where_)(
    mlx_array* res,
    const mlx_array condition,
    const mlx_array x,
    const mlx_array y,
    const mlx_stream s);
extern int (*mlx_zeros_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s);
extern int (*mlx_zeros_like_)(mlx_array* res, const mlx_array a, const mlx_stream s);
extern int (*mlx_random_bernoulli_)(
    mlx_array* res,
    const mlx_array p,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_bits_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    int width,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_categorical_shape_)(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_categorical_num_samples_)(
    mlx_array* res,
    const mlx_array logits_,
    int axis,
    int num_samples,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_categorical_)(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_gumbel_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_key_)(mlx_array* res, uint64_t seed);
extern int (*mlx_random_laplace_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_multivariate_normal_)(
    mlx_array* res,
    const mlx_array mean,
    const mlx_array cov,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_normal_broadcast_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array loc /* may be null */,
    const mlx_array scale /* may be null */,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_normal_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_permutation_)(
    mlx_array* res,
    const mlx_array x,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_permutation_arange_)(
    mlx_array* res,
    int x,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_randint_)(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_seed_)(uint64_t seed);
extern int (*mlx_random_split_num_)(
    mlx_array* res,
    const mlx_array key,
    int num,
    const mlx_stream s);
extern int (*mlx_random_split_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array key,
    const mlx_stream s);
extern int (*mlx_random_truncated_normal_)(
    mlx_array* res,
    const mlx_array lower,
    const mlx_array upper,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern int (*mlx_random_uniform_)(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
extern mlx_stream (*mlx_stream_new_)(void);
extern mlx_stream (*mlx_stream_new_device_)(mlx_device dev);
extern int (*mlx_stream_set_)(mlx_stream* stream, const mlx_stream src);
extern int (*mlx_stream_free_)(mlx_stream stream);
extern int (*mlx_stream_tostring_)(mlx_string* str, mlx_stream stream);
extern bool (*mlx_stream_equal_)(mlx_stream lhs, mlx_stream rhs);
extern int (*mlx_stream_get_device_)(mlx_device* dev, mlx_stream stream);
extern int (*mlx_stream_get_index_)(int* index, mlx_stream stream);
extern int (*mlx_synchronize_)(mlx_stream stream);
extern int (*mlx_get_default_stream_)(mlx_stream* stream, mlx_device dev);
extern int (*mlx_set_default_stream_)(mlx_stream stream);
extern mlx_stream (*mlx_default_cpu_stream_new_)(void);
extern mlx_stream (*mlx_default_gpu_stream_new_)(void);
extern mlx_string (*mlx_string_new_)(void);
extern mlx_string (*mlx_string_new_data_)(const char* str);
extern int (*mlx_string_set_)(mlx_string* str, const mlx_string src);
extern const char * (*mlx_string_data_)(mlx_string str);
extern int (*mlx_string_free_)(mlx_string str);
extern int (*mlx_detail_vmap_replace_)(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array s_inputs,
    const mlx_vector_array s_outputs,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num);
extern int (*mlx_detail_vmap_trace_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array inputs,
    const int* in_axes,
    size_t in_axes_num);
extern int (*mlx_async_eval_)(const mlx_vector_array outputs);
extern int (*mlx_checkpoint_)(mlx_closure* res, const mlx_closure fun);
extern int (*mlx_custom_function_)(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */);
extern int (*mlx_custom_vjp_)(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp);
extern int (*mlx_eval_)(const mlx_vector_array outputs);
extern int (*mlx_jvp_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents);
extern int (*mlx_value_and_grad_)(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num);
extern int (*mlx_vjp_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents);
extern mlx_vector_array (*mlx_vector_array_new_)(void);
extern int (*mlx_vector_array_set_)(mlx_vector_array* vec, const mlx_vector_array src);
extern int (*mlx_vector_array_free_)(mlx_vector_array vec);
extern mlx_vector_array (*mlx_vector_array_new_data_)(const mlx_array* data, size_t size);
extern mlx_vector_array (*mlx_vector_array_new_value_)(const mlx_array val);
extern int (*mlx_vector_array_set_data_)(
    mlx_vector_array* vec,
    const mlx_array* data,
    size_t size);
extern int (*mlx_vector_array_set_value_)(mlx_vector_array* vec, const mlx_array val);
extern int (*mlx_vector_array_append_data_)(
    mlx_vector_array vec,
    const mlx_array* data,
    size_t size);
extern int (*mlx_vector_array_append_value_)(mlx_vector_array vec, const mlx_array val);
extern size_t (*mlx_vector_array_size_)(mlx_vector_array vec);
extern int (*mlx_vector_array_get_)(
    mlx_array* res,
    const mlx_vector_array vec,
    size_t idx);
extern mlx_vector_vector_array (*mlx_vector_vector_array_new_)(void);
extern int (*mlx_vector_vector_array_set_)(
    mlx_vector_vector_array* vec,
    const mlx_vector_vector_array src);
extern int (*mlx_vector_vector_array_free_)(mlx_vector_vector_array vec);
extern mlx_vector_vector_array (*mlx_vector_vector_array_new_data_)(
    const mlx_vector_array* data,
    size_t size);
extern mlx_vector_vector_array (*mlx_vector_vector_array_new_value_)(
    const mlx_vector_array val);
extern int (*mlx_vector_vector_array_set_data_)(
    mlx_vector_vector_array* vec,
    const mlx_vector_array* data,
    size_t size);
extern int (*mlx_vector_vector_array_set_value_)(
    mlx_vector_vector_array* vec,
    const mlx_vector_array val);
extern int (*mlx_vector_vector_array_append_data_)(
    mlx_vector_vector_array vec,
    const mlx_vector_array* data,
    size_t size);
extern int (*mlx_vector_vector_array_append_value_)(
    mlx_vector_vector_array vec,
    const mlx_vector_array val);
extern size_t (*mlx_vector_vector_array_size_)(mlx_vector_vector_array vec);
extern int (*mlx_vector_vector_array_get_)(
    mlx_vector_array* res,
    const mlx_vector_vector_array vec,
    size_t idx);
extern mlx_vector_int (*mlx_vector_int_new_)(void);
extern int (*mlx_vector_int_set_)(mlx_vector_int* vec, const mlx_vector_int src);
extern int (*mlx_vector_int_free_)(mlx_vector_int vec);
extern mlx_vector_int (*mlx_vector_int_new_data_)(int* data, size_t size);
extern mlx_vector_int (*mlx_vector_int_new_value_)(int val);
extern int (*mlx_vector_int_set_data_)(mlx_vector_int* vec, int* data, size_t size);
extern int (*mlx_vector_int_set_value_)(mlx_vector_int* vec, int val);
extern int (*mlx_vector_int_append_data_)(mlx_vector_int vec, int* data, size_t size);
extern int (*mlx_vector_int_append_value_)(mlx_vector_int vec, int val);
extern size_t (*mlx_vector_int_size_)(mlx_vector_int vec);
extern int (*mlx_vector_int_get_)(int* res, const mlx_vector_int vec, size_t idx);
extern mlx_vector_string (*mlx_vector_string_new_)(void);
extern int (*mlx_vector_string_set_)(mlx_vector_string* vec, const mlx_vector_string src);
extern int (*mlx_vector_string_free_)(mlx_vector_string vec);
extern mlx_vector_string (*mlx_vector_string_new_data_)(const char** data, size_t size);
extern mlx_vector_string (*mlx_vector_string_new_value_)(const char* val);
extern int (*mlx_vector_string_set_data_)(
    mlx_vector_string* vec,
    const char** data,
    size_t size);
extern int (*mlx_vector_string_set_value_)(mlx_vector_string* vec, const char* val);
extern int (*mlx_vector_string_append_data_)(
    mlx_vector_string vec,
    const char** data,
    size_t size);
extern int (*mlx_vector_string_append_value_)(mlx_vector_string vec, const char* val);
extern size_t (*mlx_vector_string_size_)(mlx_vector_string vec);
extern int (*mlx_vector_string_get_)(char** res, const mlx_vector_string vec, size_t idx);
extern int (*mlx_version_)(mlx_string* str_);

int mlx_dynamic_load_symbols(mlx_dynamic_handle handle);

size_t mlx_dtype_size(mlx_dtype dtype);
int mlx_array_tostring(mlx_string* str, const mlx_array arr);
mlx_array mlx_array_new(void);
int mlx_array_free(mlx_array arr);
mlx_array mlx_array_new_bool(bool val);
mlx_array mlx_array_new_int(int val);
mlx_array mlx_array_new_float32(float val);
mlx_array mlx_array_new_float(float val);
mlx_array mlx_array_new_float64(double val);
mlx_array mlx_array_new_double(double val);
mlx_array mlx_array_new_complex(float real_val, float imag_val);
mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
int mlx_array_set(mlx_array* arr, const mlx_array src);
int mlx_array_set_bool(mlx_array* arr, bool val);
int mlx_array_set_int(mlx_array* arr, int val);
int mlx_array_set_float32(mlx_array* arr, float val);
int mlx_array_set_float(mlx_array* arr, float val);
int mlx_array_set_float64(mlx_array* arr, double val);
int mlx_array_set_double(mlx_array* arr, double val);
int mlx_array_set_complex(mlx_array* arr, float real_val, float imag_val);
int mlx_array_set_data(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
size_t mlx_array_itemsize(const mlx_array arr);
size_t mlx_array_size(const mlx_array arr);
size_t mlx_array_nbytes(const mlx_array arr);
size_t mlx_array_ndim(const mlx_array arr);
const int * mlx_array_shape(const mlx_array arr);
const size_t * mlx_array_strides(const mlx_array arr);
int mlx_array_dim(const mlx_array arr, int dim);
mlx_dtype mlx_array_dtype(const mlx_array arr);
int mlx_array_eval(mlx_array arr);
int mlx_array_item_bool(bool* res, const mlx_array arr);
int mlx_array_item_uint8(uint8_t* res, const mlx_array arr);
int mlx_array_item_uint16(uint16_t* res, const mlx_array arr);
int mlx_array_item_uint32(uint32_t* res, const mlx_array arr);
int mlx_array_item_uint64(uint64_t* res, const mlx_array arr);
int mlx_array_item_int8(int8_t* res, const mlx_array arr);
int mlx_array_item_int16(int16_t* res, const mlx_array arr);
int mlx_array_item_int32(int32_t* res, const mlx_array arr);
int mlx_array_item_int64(int64_t* res, const mlx_array arr);
int mlx_array_item_float32(float* res, const mlx_array arr);
int mlx_array_item_float64(double* res, const mlx_array arr);
int mlx_array_item_complex64(float _Complex* res, const mlx_array arr);
int mlx_array_item_float16(float16_t* res, const mlx_array arr);
int mlx_array_item_bfloat16(bfloat16_t* res, const mlx_array arr);
const bool * mlx_array_data_bool(const mlx_array arr);
const uint8_t * mlx_array_data_uint8(const mlx_array arr);
const uint16_t * mlx_array_data_uint16(const mlx_array arr);
const uint32_t * mlx_array_data_uint32(const mlx_array arr);
const uint64_t * mlx_array_data_uint64(const mlx_array arr);
const int8_t * mlx_array_data_int8(const mlx_array arr);
const int16_t * mlx_array_data_int16(const mlx_array arr);
const int32_t * mlx_array_data_int32(const mlx_array arr);
const int64_t * mlx_array_data_int64(const mlx_array arr);
const float * mlx_array_data_float32(const mlx_array arr);
const double * mlx_array_data_float64(const mlx_array arr);
const float _Complex * mlx_array_data_complex64(const mlx_array arr);
const float16_t * mlx_array_data_float16(const mlx_array arr);
const bfloat16_t * mlx_array_data_bfloat16(const mlx_array arr);
int _mlx_array_is_available(bool* res, const mlx_array arr);
int _mlx_array_wait(const mlx_array arr);
int _mlx_array_is_contiguous(bool* res, const mlx_array arr);
int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr);
int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr);
mlx_closure mlx_closure_new(void);
int mlx_closure_free(mlx_closure cls);
mlx_closure mlx_closure_new_func(
    int (*fun)(mlx_vector_array*, const mlx_vector_array));
mlx_closure mlx_closure_new_func_payload(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_set(mlx_closure* cls, const mlx_closure src);
int mlx_closure_apply(
    mlx_vector_array* res,
    mlx_closure cls,
    const mlx_vector_array input);
mlx_closure mlx_closure_new_unary(int (*fun)(mlx_array*, const mlx_array));
mlx_closure_kwargs mlx_closure_kwargs_new(void);
int mlx_closure_kwargs_free(mlx_closure_kwargs cls);
mlx_closure_kwargs mlx_closure_kwargs_new_func(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_map_string_to_array));
mlx_closure_kwargs mlx_closure_kwargs_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_kwargs_set(
    mlx_closure_kwargs* cls,
    const mlx_closure_kwargs src);
int mlx_closure_kwargs_apply(
    mlx_vector_array* res,
    mlx_closure_kwargs cls,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1);
mlx_closure_value_and_grad mlx_closure_value_and_grad_new(void);
int mlx_closure_value_and_grad_free(mlx_closure_value_and_grad cls);
mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func(
    int (*fun)(mlx_vector_array*, mlx_vector_array*, const mlx_vector_array));
mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_array*,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_value_and_grad_set(
    mlx_closure_value_and_grad* cls,
    const mlx_closure_value_and_grad src);
int mlx_closure_value_and_grad_apply(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    mlx_closure_value_and_grad cls,
    const mlx_vector_array input);
mlx_closure_custom mlx_closure_custom_new(void);
int mlx_closure_custom_free(mlx_closure_custom cls);
mlx_closure_custom mlx_closure_custom_new_func(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const mlx_vector_array));
mlx_closure_custom mlx_closure_custom_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_custom_set(
    mlx_closure_custom* cls,
    const mlx_closure_custom src);
int mlx_closure_custom_apply(
    mlx_vector_array* res,
    mlx_closure_custom cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2);
mlx_closure_custom_jvp mlx_closure_custom_jvp_new(void);
int mlx_closure_custom_jvp_free(mlx_closure_custom_jvp cls);
mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const int*,
    size_t _num));
mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_custom_jvp_set(
    mlx_closure_custom_jvp* cls,
    const mlx_closure_custom_jvp src);
int mlx_closure_custom_jvp_apply(
    mlx_vector_array* res,
    mlx_closure_custom_jvp cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const int* input_2,
    size_t input_2_num);
mlx_closure_custom_vmap mlx_closure_custom_vmap_new(void);
int mlx_closure_custom_vmap_free(mlx_closure_custom_vmap cls);
mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func(int (*fun)(
    mlx_vector_array*,
    mlx_vector_int*,
    const mlx_vector_array,
    const int*,
    size_t _num));
mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*));
int mlx_closure_custom_vmap_set(
    mlx_closure_custom_vmap* cls,
    const mlx_closure_custom_vmap src);
int mlx_closure_custom_vmap_apply(
    mlx_vector_array* res_0,
    mlx_vector_int* res_1,
    mlx_closure_custom_vmap cls,
    const mlx_vector_array input_0,
    const int* input_1,
    size_t input_1_num);
int mlx_compile(mlx_closure* res, const mlx_closure fun, bool shapeless);
int mlx_detail_compile(
    mlx_closure* res,
    const mlx_closure fun,
    uintptr_t fun_id,
    bool shapeless,
    const uint64_t* constants,
    size_t constants_num);
int mlx_detail_compile_clear_cache(void);
int mlx_detail_compile_erase(uintptr_t fun_id);
int mlx_disable_compile(void);
int mlx_enable_compile(void);
int mlx_set_compile_mode(mlx_compile_mode mode);
mlx_device mlx_device_new(void);
mlx_device mlx_device_new_type(mlx_device_type type, int index);
int mlx_device_free(mlx_device dev);
int mlx_device_set(mlx_device* dev, const mlx_device src);
int mlx_device_tostring(mlx_string* str, mlx_device dev);
bool mlx_device_equal(mlx_device lhs, mlx_device rhs);
int mlx_device_get_index(int* index, mlx_device dev);
int mlx_device_get_type(mlx_device_type* type, mlx_device dev);
int mlx_get_default_device(mlx_device* dev);
int mlx_set_default_device(mlx_device dev);
int mlx_distributed_group_rank(mlx_distributed_group group);
int mlx_distributed_group_size(mlx_distributed_group group);
mlx_distributed_group mlx_distributed_group_split(mlx_distributed_group group, int color, int key);
bool mlx_distributed_is_available(void);
mlx_distributed_group mlx_distributed_init(bool strict);
int mlx_distributed_all_gather(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream S);
int mlx_distributed_all_max(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
int mlx_distributed_all_min(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
int mlx_distributed_all_sum(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
int mlx_distributed_recv(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
int mlx_distributed_recv_like(
    mlx_array* res,
    const mlx_array x,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
int mlx_distributed_send(
    mlx_array* res,
    const mlx_array x,
    int dst,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
int mlx_distributed_sum_scatter(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);
void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*));
void _mlx_error(const char* file, const int line, const char* fmt, ...);
int mlx_export_function(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless);
int mlx_export_function_kwargs(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless);
mlx_function_exporter mlx_function_exporter_new(
    const char* file,
    const mlx_closure fun,
    bool shapeless);
int mlx_function_exporter_free(mlx_function_exporter xfunc);
int mlx_function_exporter_apply(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args);
int mlx_function_exporter_apply_kwargs(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);
mlx_imported_function mlx_imported_function_new(const char* file);
int mlx_imported_function_free(mlx_imported_function xfunc);
int mlx_imported_function_apply(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args);
int mlx_imported_function_apply_kwargs(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);
int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);
mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);
void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);
int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);
int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);
int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);
mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);
void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);
int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);
int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s);
int mlx_fft_fft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_fft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_fftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_fftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_ifft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_ifft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_ifftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_ifftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_irfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_irfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_irfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_rfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_rfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_rfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
mlx_io_reader mlx_io_reader_new(void* desc, mlx_io_vtable vtable);
int mlx_io_reader_descriptor(void** desc_, mlx_io_reader io);
int mlx_io_reader_tostring(mlx_string* str_, mlx_io_reader io);
int mlx_io_reader_free(mlx_io_reader io);
mlx_io_writer mlx_io_writer_new(void* desc, mlx_io_vtable vtable);
int mlx_io_writer_descriptor(void** desc_, mlx_io_writer io);
int mlx_io_writer_tostring(mlx_string* str_, mlx_io_writer io);
int mlx_io_writer_free(mlx_io_writer io);
int mlx_load_reader(
    mlx_array* res,
    mlx_io_reader in_stream,
    const mlx_stream s);
int mlx_load(mlx_array* res, const char* file, const mlx_stream s);
int mlx_load_safetensors_reader(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    mlx_io_reader in_stream,
    const mlx_stream s);
int mlx_load_safetensors(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    const char* file,
    const mlx_stream s);
int mlx_save_writer(mlx_io_writer out_stream, const mlx_array a);
int mlx_save(const char* file, const mlx_array a);
int mlx_save_safetensors_writer(
    mlx_io_writer in_stream,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata);
int mlx_save_safetensors(
    const char* file,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata);
int mlx_linalg_cholesky(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
int mlx_linalg_cholesky_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
int mlx_linalg_cross(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s);
int mlx_linalg_eig(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
int mlx_linalg_eigh(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s);
int mlx_linalg_eigvals(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_eigvalsh(
    mlx_array* res,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s);
int mlx_linalg_inv(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_lu(mlx_vector_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_lu_factor(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
int mlx_linalg_norm(
    mlx_array* res,
    const mlx_array a,
    double ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
int mlx_linalg_norm_matrix(
    mlx_array* res,
    const mlx_array a,
    const char* ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
int mlx_linalg_norm_l2(
    mlx_array* res,
    const mlx_array a,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
int mlx_linalg_pinv(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_qr(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
int mlx_linalg_solve(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_linalg_solve_triangular(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool upper,
    const mlx_stream s);
int mlx_linalg_svd(
    mlx_vector_array* res,
    const mlx_array a,
    bool compute_uv,
    const mlx_stream s);
int mlx_linalg_tri_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
mlx_map_string_to_array mlx_map_string_to_array_new(void);
int mlx_map_string_to_array_set(
    mlx_map_string_to_array* map,
    const mlx_map_string_to_array src);
int mlx_map_string_to_array_free(mlx_map_string_to_array map);
int mlx_map_string_to_array_insert(
    mlx_map_string_to_array map,
    const char* key,
    const mlx_array value);
int mlx_map_string_to_array_get(
    mlx_array* value,
    const mlx_map_string_to_array map,
    const char* key);
mlx_map_string_to_array_iterator mlx_map_string_to_array_iterator_new(
    mlx_map_string_to_array map);
int mlx_map_string_to_array_iterator_free(mlx_map_string_to_array_iterator it);
int mlx_map_string_to_array_iterator_next(
    const char** key,
    mlx_array* value,
    mlx_map_string_to_array_iterator it);
mlx_map_string_to_string mlx_map_string_to_string_new(void);
int mlx_map_string_to_string_set(
    mlx_map_string_to_string* map,
    const mlx_map_string_to_string src);
int mlx_map_string_to_string_free(mlx_map_string_to_string map);
int mlx_map_string_to_string_insert(
    mlx_map_string_to_string map,
    const char* key,
    const char* value);
int mlx_map_string_to_string_get(
    const char** value,
    const mlx_map_string_to_string map,
    const char* key);
mlx_map_string_to_string_iterator mlx_map_string_to_string_iterator_new(
    mlx_map_string_to_string map);
int mlx_map_string_to_string_iterator_free(
    mlx_map_string_to_string_iterator it);
int mlx_map_string_to_string_iterator_next(
    const char** key,
    const char** value,
    mlx_map_string_to_string_iterator it);
int mlx_clear_cache(void);
int mlx_get_active_memory(size_t* res);
int mlx_get_cache_memory(size_t* res);
int mlx_get_memory_limit(size_t* res);
int mlx_get_peak_memory(size_t* res);
int mlx_reset_peak_memory(void);
int mlx_set_cache_limit(size_t* res, size_t limit);
int mlx_set_memory_limit(size_t* res, size_t limit);
int mlx_set_wired_limit(size_t* res, size_t limit);
mlx_metal_device_info_t mlx_metal_device_info(void);
int mlx_metal_is_available(bool* res);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);
int mlx_abs(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_add(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_addmm(
    mlx_array* res,
    const mlx_array c,
    const mlx_array a,
    const mlx_array b,
    float alpha,
    float beta,
    const mlx_stream s);
int mlx_all_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_all_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_all(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_allclose(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s);
int mlx_any_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_any_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_any(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_arange(
    mlx_array* res,
    double start,
    double stop,
    double step,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_arccos(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_arccosh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_arcsin(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_arcsinh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_arctan(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_arctan2(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_arctanh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_argmax_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_argmax(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_argmin_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_argmin(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_argpartition_axis(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s);
int mlx_argpartition(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s);
int mlx_argsort_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
int mlx_argsort(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_array_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool equal_nan,
    const mlx_stream s);
int mlx_as_strided(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const int64_t* strides,
    size_t strides_num,
    size_t offset,
    const mlx_stream s);
int mlx_astype(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_atleast_1d(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_atleast_2d(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_atleast_3d(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_bitwise_and(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_bitwise_invert(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_bitwise_or(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_bitwise_xor(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_block_masked_mm(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int block_size,
    const mlx_array mask_out /* may be null */,
    const mlx_array mask_lhs /* may be null */,
    const mlx_array mask_rhs /* may be null */,
    const mlx_stream s);
int mlx_broadcast_arrays(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_stream s);
int mlx_broadcast_to(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s);
int mlx_ceil(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_clip(
    mlx_array* res,
    const mlx_array a,
    const mlx_array a_min /* may be null */,
    const mlx_array a_max /* may be null */,
    const mlx_stream s);
int mlx_concatenate_axis(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s);
int mlx_concatenate(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s);
int mlx_conjugate(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_contiguous(
    mlx_array* res,
    const mlx_array a,
    bool allow_col_major,
    const mlx_stream s);
int mlx_conv1d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int groups,
    const mlx_stream s);
int mlx_conv2d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int padding_0,
    int padding_1,
    int dilation_0,
    int dilation_1,
    int groups,
    const mlx_stream s);
int mlx_conv3d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int stride_2,
    int padding_0,
    int padding_1,
    int padding_2,
    int dilation_0,
    int dilation_1,
    int dilation_2,
    int groups,
    const mlx_stream s);
int mlx_conv_general(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    const int* stride,
    size_t stride_num,
    const int* padding_lo,
    size_t padding_lo_num,
    const int* padding_hi,
    size_t padding_hi_num,
    const int* kernel_dilation,
    size_t kernel_dilation_num,
    const int* input_dilation,
    size_t input_dilation_num,
    int groups,
    bool flip,
    const mlx_stream s);
int mlx_conv_transpose1d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int output_padding,
    int groups,
    const mlx_stream s);
int mlx_conv_transpose2d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int padding_0,
    int padding_1,
    int dilation_0,
    int dilation_1,
    int output_padding_0,
    int output_padding_1,
    int groups,
    const mlx_stream s);
int mlx_conv_transpose3d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int stride_2,
    int padding_0,
    int padding_1,
    int padding_2,
    int dilation_0,
    int dilation_1,
    int dilation_2,
    int output_padding_0,
    int output_padding_1,
    int output_padding_2,
    int groups,
    const mlx_stream s);
int mlx_copy(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_cos(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_cosh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_cummax(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
int mlx_cummin(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
int mlx_cumprod(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
int mlx_cumsum(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
int mlx_degrees(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_depends(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array dependencies);
int mlx_dequantize(
    mlx_array* res,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    mlx_optional_dtype dtype,
    const mlx_stream s);
int mlx_diag(mlx_array* res, const mlx_array a, int k, const mlx_stream s);
int mlx_diagonal(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    const mlx_stream s);
int mlx_divide(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_divmod(
    mlx_vector_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_einsum(
    mlx_array* res,
    const char* subscripts,
    const mlx_vector_array operands,
    const mlx_stream s);
int mlx_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_erf(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_erfinv(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_exp(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_expand_dims_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_expand_dims(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
int mlx_expm1(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_eye(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_flatten(
    mlx_array* res,
    const mlx_array a,
    int start_axis,
    int end_axis,
    const mlx_stream s);
int mlx_floor(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_floor_divide(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_from_fp8(
    mlx_array* res,
    const mlx_array x,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_full(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_full_like(
    mlx_array* res,
    const mlx_array a,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_gather(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const int* axes,
    size_t axes_num,
    const int* slice_sizes,
    size_t slice_sizes_num,
    const mlx_stream s);
int mlx_gather_mm(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool sorted_indices,
    const mlx_stream s);
int mlx_gather_qmm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    bool sorted_indices,
    const mlx_stream s);
int mlx_greater(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_greater_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_hadamard_transform(
    mlx_array* res,
    const mlx_array a,
    mlx_optional_float scale,
    const mlx_stream s);
int mlx_identity(mlx_array* res, int n, mlx_dtype dtype, const mlx_stream s);
int mlx_imag(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_inner(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_isclose(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s);
int mlx_isfinite(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_isinf(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_isnan(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_isneginf(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_isposinf(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_kron(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_left_shift(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_less(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_less_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_linspace(
    mlx_array* res,
    double start,
    double stop,
    int num,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_log(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_log10(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_log1p(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_log2(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_logaddexp(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_logcumsumexp(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s);
int mlx_logical_and(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_logical_not(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_logical_or(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_logsumexp_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_logsumexp_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_logsumexp(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_masked_scatter(
    mlx_array* res,
    const mlx_array a,
    const mlx_array mask,
    const mlx_array src,
    const mlx_stream s);
int mlx_matmul(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_max_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_max_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_max(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_maximum(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_mean_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_mean_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_mean(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_median(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_meshgrid(
    mlx_vector_array* res,
    const mlx_vector_array arrays,
    bool sparse,
    const char* indexing,
    const mlx_stream s);
int mlx_min_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_min_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_min(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_minimum(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_moveaxis(
    mlx_array* res,
    const mlx_array a,
    int source,
    int destination,
    const mlx_stream s);
int mlx_multiply(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_nan_to_num(
    mlx_array* res,
    const mlx_array a,
    float nan,
    mlx_optional_float posinf,
    mlx_optional_float neginf,
    const mlx_stream s);
int mlx_negative(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_not_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_number_of_elements(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool inverted,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_ones(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_ones_like(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_outer(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_pad(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const int* low_pad_size,
    size_t low_pad_size_num,
    const int* high_pad_size,
    size_t high_pad_size_num,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s);
int mlx_pad_symmetric(
    mlx_array* res,
    const mlx_array a,
    int pad_width,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s);
int mlx_partition_axis(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s);
int mlx_partition(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s);
int mlx_power(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_prod_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_prod_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_prod(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_put_along_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s);
int mlx_quantize(
    mlx_vector_array* res,
    const mlx_array w,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s);
int mlx_quantized_matmul(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s);
int mlx_radians(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_real(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_reciprocal(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_remainder(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_repeat_axis(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    int axis,
    const mlx_stream s);
int mlx_repeat(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    const mlx_stream s);
int mlx_reshape(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s);
int mlx_right_shift(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_roll_axis(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    int axis,
    const mlx_stream s);
int mlx_roll_axes(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_roll(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const mlx_stream s);
int mlx_round(
    mlx_array* res,
    const mlx_array a,
    int decimals,
    const mlx_stream s);
int mlx_rsqrt(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_scatter(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_scatter_add(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_scatter_add_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s);
int mlx_scatter_max(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_scatter_min(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_scatter_prod(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_segmented_mm(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array segments,
    const mlx_stream s);
int mlx_sigmoid(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_sign(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_sin(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_sinh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_slice(
    mlx_array* res,
    const mlx_array a,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s);
int mlx_slice_dynamic(
    mlx_array* res,
    const mlx_array a,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const int* slice_size,
    size_t slice_size_num,
    const mlx_stream s);
int mlx_slice_update(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s);
int mlx_slice_update_dynamic(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_softmax_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool precise,
    const mlx_stream s);
int mlx_softmax_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool precise,
    const mlx_stream s);
int mlx_softmax(
    mlx_array* res,
    const mlx_array a,
    bool precise,
    const mlx_stream s);
int mlx_sort_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
int mlx_sort(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_split(
    mlx_vector_array* res,
    const mlx_array a,
    int num_splits,
    int axis,
    const mlx_stream s);
int mlx_split_sections(
    mlx_vector_array* res,
    const mlx_array a,
    const int* indices,
    size_t indices_num,
    int axis,
    const mlx_stream s);
int mlx_sqrt(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_square(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_squeeze_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_squeeze_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s);
int mlx_squeeze(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_stack_axis(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s);
int mlx_stack(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s);
int mlx_std_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s);
int mlx_std_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s);
int mlx_std(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s);
int mlx_stop_gradient(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_subtract(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_sum_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);
int mlx_sum_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);
int mlx_sum(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s);
int mlx_swapaxes(
    mlx_array* res,
    const mlx_array a,
    int axis1,
    int axis2,
    const mlx_stream s);
int mlx_take_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s);
int mlx_take(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_stream s);
int mlx_take_along_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s);
int mlx_tan(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_tanh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_tensordot(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const int* axes_a,
    size_t axes_a_num,
    const int* axes_b,
    size_t axes_b_num,
    const mlx_stream s);
int mlx_tensordot_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s);
int mlx_tile(
    mlx_array* res,
    const mlx_array arr,
    const int* reps,
    size_t reps_num,
    const mlx_stream s);
int mlx_to_fp8(mlx_array* res, const mlx_array x, const mlx_stream s);
int mlx_topk_axis(
    mlx_array* res,
    const mlx_array a,
    int k,
    int axis,
    const mlx_stream s);
int mlx_topk(mlx_array* res, const mlx_array a, int k, const mlx_stream s);
int mlx_trace(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_transpose_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_transpose(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_tri(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype type,
    const mlx_stream s);
int mlx_tril(mlx_array* res, const mlx_array x, int k, const mlx_stream s);
int mlx_triu(mlx_array* res, const mlx_array x, int k, const mlx_stream s);
int mlx_unflatten(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_stream s);
int mlx_var_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s);
int mlx_var_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s);
int mlx_var(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s);
int mlx_view(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_where(
    mlx_array* res,
    const mlx_array condition,
    const mlx_array x,
    const mlx_array y,
    const mlx_stream s);
int mlx_zeros(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s);
int mlx_zeros_like(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_random_bernoulli(
    mlx_array* res,
    const mlx_array p,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_bits(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    int width,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_categorical_shape(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_categorical_num_samples(
    mlx_array* res,
    const mlx_array logits_,
    int axis,
    int num_samples,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_categorical(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_gumbel(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_key(mlx_array* res, uint64_t seed);
int mlx_random_laplace(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_multivariate_normal(
    mlx_array* res,
    const mlx_array mean,
    const mlx_array cov,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_normal_broadcast(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array loc /* may be null */,
    const mlx_array scale /* may be null */,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_normal(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_permutation(
    mlx_array* res,
    const mlx_array x,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_permutation_arange(
    mlx_array* res,
    int x,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_randint(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_seed(uint64_t seed);
int mlx_random_split_num(
    mlx_array* res,
    const mlx_array key,
    int num,
    const mlx_stream s);
int mlx_random_split(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array key,
    const mlx_stream s);
int mlx_random_truncated_normal(
    mlx_array* res,
    const mlx_array lower,
    const mlx_array upper,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
int mlx_random_uniform(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s);
mlx_stream mlx_stream_new(void);
mlx_stream mlx_stream_new_device(mlx_device dev);
int mlx_stream_set(mlx_stream* stream, const mlx_stream src);
int mlx_stream_free(mlx_stream stream);
int mlx_stream_tostring(mlx_string* str, mlx_stream stream);
bool mlx_stream_equal(mlx_stream lhs, mlx_stream rhs);
int mlx_stream_get_device(mlx_device* dev, mlx_stream stream);
int mlx_stream_get_index(int* index, mlx_stream stream);
int mlx_synchronize(mlx_stream stream);
int mlx_get_default_stream(mlx_stream* stream, mlx_device dev);
int mlx_set_default_stream(mlx_stream stream);
mlx_stream mlx_default_cpu_stream_new(void);
mlx_stream mlx_default_gpu_stream_new(void);
mlx_string mlx_string_new(void);
mlx_string mlx_string_new_data(const char* str);
int mlx_string_set(mlx_string* str, const mlx_string src);
const char * mlx_string_data(mlx_string str);
int mlx_string_free(mlx_string str);
int mlx_detail_vmap_replace(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array s_inputs,
    const mlx_vector_array s_outputs,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num);
int mlx_detail_vmap_trace(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array inputs,
    const int* in_axes,
    size_t in_axes_num);
int mlx_async_eval(const mlx_vector_array outputs);
int mlx_checkpoint(mlx_closure* res, const mlx_closure fun);
int mlx_custom_function(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */);
int mlx_custom_vjp(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp);
int mlx_eval(const mlx_vector_array outputs);
int mlx_jvp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents);
int mlx_value_and_grad(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num);
int mlx_vjp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents);
mlx_vector_array mlx_vector_array_new(void);
int mlx_vector_array_set(mlx_vector_array* vec, const mlx_vector_array src);
int mlx_vector_array_free(mlx_vector_array vec);
mlx_vector_array mlx_vector_array_new_data(const mlx_array* data, size_t size);
mlx_vector_array mlx_vector_array_new_value(const mlx_array val);
int mlx_vector_array_set_data(
    mlx_vector_array* vec,
    const mlx_array* data,
    size_t size);
int mlx_vector_array_set_value(mlx_vector_array* vec, const mlx_array val);
int mlx_vector_array_append_data(
    mlx_vector_array vec,
    const mlx_array* data,
    size_t size);
int mlx_vector_array_append_value(mlx_vector_array vec, const mlx_array val);
size_t mlx_vector_array_size(mlx_vector_array vec);
int mlx_vector_array_get(
    mlx_array* res,
    const mlx_vector_array vec,
    size_t idx);
mlx_vector_vector_array mlx_vector_vector_array_new(void);
int mlx_vector_vector_array_set(
    mlx_vector_vector_array* vec,
    const mlx_vector_vector_array src);
int mlx_vector_vector_array_free(mlx_vector_vector_array vec);
mlx_vector_vector_array mlx_vector_vector_array_new_data(
    const mlx_vector_array* data,
    size_t size);
mlx_vector_vector_array mlx_vector_vector_array_new_value(
    const mlx_vector_array val);
int mlx_vector_vector_array_set_data(
    mlx_vector_vector_array* vec,
    const mlx_vector_array* data,
    size_t size);
int mlx_vector_vector_array_set_value(
    mlx_vector_vector_array* vec,
    const mlx_vector_array val);
int mlx_vector_vector_array_append_data(
    mlx_vector_vector_array vec,
    const mlx_vector_array* data,
    size_t size);
int mlx_vector_vector_array_append_value(
    mlx_vector_vector_array vec,
    const mlx_vector_array val);
size_t mlx_vector_vector_array_size(mlx_vector_vector_array vec);
int mlx_vector_vector_array_get(
    mlx_vector_array* res,
    const mlx_vector_vector_array vec,
    size_t idx);
mlx_vector_int mlx_vector_int_new(void);
int mlx_vector_int_set(mlx_vector_int* vec, const mlx_vector_int src);
int mlx_vector_int_free(mlx_vector_int vec);
mlx_vector_int mlx_vector_int_new_data(int* data, size_t size);
mlx_vector_int mlx_vector_int_new_value(int val);
int mlx_vector_int_set_data(mlx_vector_int* vec, int* data, size_t size);
int mlx_vector_int_set_value(mlx_vector_int* vec, int val);
int mlx_vector_int_append_data(mlx_vector_int vec, int* data, size_t size);
int mlx_vector_int_append_value(mlx_vector_int vec, int val);
size_t mlx_vector_int_size(mlx_vector_int vec);
int mlx_vector_int_get(int* res, const mlx_vector_int vec, size_t idx);
mlx_vector_string mlx_vector_string_new(void);
int mlx_vector_string_set(mlx_vector_string* vec, const mlx_vector_string src);
int mlx_vector_string_free(mlx_vector_string vec);
mlx_vector_string mlx_vector_string_new_data(const char** data, size_t size);
mlx_vector_string mlx_vector_string_new_value(const char* val);
int mlx_vector_string_set_data(
    mlx_vector_string* vec,
    const char** data,
    size_t size);
int mlx_vector_string_set_value(mlx_vector_string* vec, const char* val);
int mlx_vector_string_append_data(
    mlx_vector_string vec,
    const char** data,
    size_t size);
int mlx_vector_string_append_value(mlx_vector_string vec, const char* val);
size_t mlx_vector_string_size(mlx_vector_string vec);
int mlx_vector_string_get(char** res, const mlx_vector_string vec, size_t idx);
int mlx_version(mlx_string* str_);

#endif // MLX_GENERATED_H
